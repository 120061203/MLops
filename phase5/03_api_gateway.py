# ============================================================
# 03_api_gateway.py
# 主題：API Gateway + Lambda — 把 Endpoint 包成公開 API
#
# 問題：SageMaker Endpoint 需要 AWS 認證才能呼叫，
#       一般使用者（或 Postman）沒有 AWS 帳號沒辦法用。
#
# 解法：
#   Postman/任何人
#     → API Gateway（公開 HTTPS，無需 AWS 認證）
#     → Lambda（有 AWS 認證，幫你轉發請求）
#     → SageMaker Endpoint
#
# 這是真實產品的標準架構。
# ============================================================

import json
import boto3
import zipfile
import os

with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region        = config["region"]
role          = config["role"]
endpoint_name = config["endpoint_name"]
account_id    = config["account_id"]

lambda_client  = boto3.client("lambda",      region_name=region)
apigw_client   = boto3.client("apigateway",  region_name=region)
iam_client     = boto3.client("iam")

print("=== API Gateway + Lambda 設定 ===\n")

# ─────────────────────────────────────────
# 1. 建立 Lambda 函式程式碼
# ─────────────────────────────────────────
print("=== 1. 建立 Lambda 函式 ===")

lambda_code = f'''
import json
import boto3

ENDPOINT_NAME = "{endpoint_name}"
REGION        = "{region}"

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

def lambda_handler(event, context):
    try:
        # 解析請求
        body = json.loads(event.get("body", "{{}}"))
        texts = body.get("texts", [])

        if not texts:
            return {{
                "statusCode": 400,
                "headers": {{"Content-Type": "application/json"}},
                "body": json.dumps({{"error": "texts 欄位不能為空"}}),
            }}

        # 呼叫 SageMaker Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps({{"texts": texts}}),
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        if isinstance(result, list) and isinstance(result[0], str):
            result = json.loads(result[0])

        return {{
            "statusCode": 200,
            "headers": {{
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",   # 允許 CORS
            }},
            "body": json.dumps(result, ensure_ascii=False),
        }}

    except Exception as e:
        return {{
            "statusCode": 500,
            "headers": {{"Content-Type": "application/json"}},
            "body": json.dumps({{"error": str(e)}}),
        }}
'''

# 打包成 zip
os.makedirs("./phase5/lambda", exist_ok=True)
with open("./phase5/lambda/lambda_function.py", "w") as f:
    f.write(lambda_code)

zip_path = "./phase5/lambda/lambda_function.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write("./phase5/lambda/lambda_function.py", "lambda_function.py")

print("Lambda 程式碼打包完成")

# ─────────────────────────────────────────
# 2. 建立 Lambda IAM Role
# ─────────────────────────────────────────
print("\n=== 2. 建立 Lambda IAM Role ===")

LAMBDA_ROLE_NAME = "mlops-lambda-sagemaker-role"

trust_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }]
}

try:
    lambda_role = iam_client.create_role(
        RoleName=LAMBDA_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Lambda role for SageMaker endpoint invocation",
    )
    lambda_role_arn = lambda_role["Role"]["Arn"]

    # 附加 Policy
    for policy in [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    ]:
        iam_client.attach_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyArn=policy)

    print(f"Lambda Role 建立：{lambda_role_arn}")

    import time
    print("等待 IAM role 生效（10 秒）...")
    time.sleep(10)

except iam_client.exceptions.EntityAlreadyExistsException:
    lambda_role_arn = f"arn:aws:iam::{account_id}:role/{LAMBDA_ROLE_NAME}"
    print(f"Lambda Role 已存在：{lambda_role_arn}")

# ─────────────────────────────────────────
# 3. 部署 Lambda 函式
# ─────────────────────────────────────────
print("\n=== 3. 部署 Lambda 函式 ===")

FUNCTION_NAME = "mlops-sentiment-api"

with open(zip_path, "rb") as f:
    zip_bytes = f.read()

try:
    fn = lambda_client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.11",
        Role=lambda_role_arn,
        Handler="lambda_function.lambda_handler",
        Code={"ZipFile": zip_bytes},
        Timeout=30,
        MemorySize=256,
    )
    function_arn = fn["FunctionArn"]
    print(f"Lambda 函式建立：{function_arn}")

except lambda_client.exceptions.ResourceConflictException:
    fn = lambda_client.update_function_code(
        FunctionName=FUNCTION_NAME,
        ZipFile=zip_bytes,
    )
    function_arn = fn["FunctionArn"]
    print(f"Lambda 函式更新：{function_arn}")

# ─────────────────────────────────────────
# 4. 建立 API Gateway
# ─────────────────────────────────────────
print("\n=== 4. 建立 API Gateway ===")

API_NAME = "mlops-sentiment-api"

# 建立 REST API
api = apigw_client.create_rest_api(
    name=API_NAME,
    description="MLOps 情感分類 API",
    endpointConfiguration={"types": ["REGIONAL"]},
)
api_id = api["id"]
print(f"API ID: {api_id}")

# 取得 root resource
resources = apigw_client.get_resources(restApiId=api_id)
root_id   = resources["items"][0]["id"]

# 建立 /predict resource
resource = apigw_client.create_resource(
    restApiId=api_id,
    parentId=root_id,
    pathPart="predict",
)
resource_id = resource["id"]

# 建立 POST method
apigw_client.put_method(
    restApiId=api_id,
    resourceId=resource_id,
    httpMethod="POST",
    authorizationType="NONE",   # 公開，不需要 AWS 認證
)

# 整合 Lambda
apigw_client.put_integration(
    restApiId=api_id,
    resourceId=resource_id,
    httpMethod="POST",
    type="AWS_PROXY",
    integrationHttpMethod="POST",
    uri=f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{function_arn}/invocations",
)

# 給 API Gateway 呼叫 Lambda 的權限
lambda_client.add_permission(
    FunctionName=FUNCTION_NAME,
    StatementId="apigateway-invoke",
    Action="lambda:InvokeFunction",
    Principal="apigateway.amazonaws.com",
    SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/POST/predict",
)

# 部署
apigw_client.create_deployment(
    restApiId=api_id,
    stageName="prod",
)

api_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/predict"
print(f"\nAPI 部署完成！")
print(f"URL: {api_url}")

# ─────────────────────────────────────────
# 5. 更新設定
# ─────────────────────────────────────────
config["api_gateway_url"] = api_url
config["lambda_function"]  = FUNCTION_NAME
config["api_id"]           = api_id

with open("./phase4/aws_config.json", "w") as f:
    json.dump(config, f, indent=2)

# ─────────────────────────────────────────
# 6. 測試公開 API
# ─────────────────────────────────────────
print("\n=== 5. 測試公開 API ===")
print(f"用 curl 或 Postman 測試（不需要 AWS 認證）：\n")
print(f"curl -X POST {api_url} \\")
print(f"  -H 'Content-Type: application/json' \\")
print(f"  -d '{{\"texts\": [\"This movie was absolutely amazing!\"]}}'")

import urllib.request

req = urllib.request.Request(
    api_url,
    data=json.dumps({"texts": ["This movie was absolutely amazing!"]}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())
    print(f"\n測試結果: {result}")

print("""
=== 重點整理 ===

架構：
  使用者/Postman
    → API Gateway（/prod/predict，公開 HTTPS）
    → Lambda（帶 AWS 認證轉發請求）
    → SageMaker Endpoint

優點：
  - 使用者不需要有 AWS 帳號
  - 可以加 Rate Limiting（防止被打爆）
  - 可以加 API Key 做存取控制
  - 可以加 WAF 防止惡意請求

Postman 測試方式：
  POST {api_url}
  Header: Content-Type: application/json
  Body: {{"texts": ["your text here"]}}
  （不需要設定 AWS Signature！）

下一步（04a/04b）：
  Bitbucket / Jenkins CI/CD 自動觸發整條 pipeline
""")

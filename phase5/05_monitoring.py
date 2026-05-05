# ============================================================
# 05_monitoring.py
# 主題：CloudWatch 監控 SageMaker Endpoint
#
# 模型上線後需要持續監控：
#   - 延遲（Latency）：使用者等待時間
#   - 錯誤率（Error Rate）：500 錯誤比例
#   - 流量（Invocations）：每分鐘呼叫次數
#   - Model Drift：輸入資料分布是否改變（需要 SageMaker Clarify）
#
# 這個檔案：
#   1. 建立 CloudWatch Dashboard
#   2. 設定警報（延遲過高 / 錯誤率過高 自動通知）
#   3. 查看 Endpoint 即時指標
# ============================================================

import json
import boto3

with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region        = config["region"]
endpoint_name = config["endpoint_name"]
account_id    = config["account_id"]

cw = boto3.client("cloudwatch", region_name=region)
sm = boto3.client("sagemaker",  region_name=region)

print("=== CloudWatch 監控設定 ===\n")

# ─────────────────────────────────────────
# 1. 建立 SNS Topic（警報通知用）
# ─────────────────────────────────────────
print("=== 1. 建立 SNS 通知 Topic ===")

sns = boto3.client("sns", region_name=region)

try:
    topic = sns.create_topic(Name="mlops-alerts")
    topic_arn = topic["TopicArn"]
    print(f"SNS Topic: {topic_arn}")

    # 訂閱 Email（收到警報時寄信）
    # 需要先到信箱確認訂閱
    EMAIL = "your-email@example.com"   # ← 改成你的 email
    sns.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=EMAIL,
    )
    print(f"Email 訂閱已送出：{EMAIL}（請到信箱確認）")

except Exception as e:
    print(f"SNS 設定：{e}")
    topic_arn = f"arn:aws:sns:{region}:{account_id}:mlops-alerts"

# ─────────────────────────────────────────
# 2. 設定 CloudWatch 警報
# ─────────────────────────────────────────
print("\n=== 2. 設定警報 ===")

# SageMaker Endpoint 的 CloudWatch 指標：
# Namespace: AWS/SageMaker
# Dimensions: EndpointName, VariantName

VARIANT = "AllTraffic"

alarms = [
    {
        "name":        f"{endpoint_name}-high-latency",
        "description": "推論延遲過高（P99 > 2秒）",
        "metric":      "ModelLatency",
        "threshold":   2000000,   # 單位：微秒（2秒 = 2,000,000 微秒）
        "comparison":  "GreaterThanThreshold",
        "statistic":   "p99",
    },
    {
        "name":        f"{endpoint_name}-high-error-rate",
        "description": "5xx 錯誤率過高（> 5%）",
        "metric":      "Invocation5XXErrors",
        "threshold":   5,
        "comparison":  "GreaterThanThreshold",
        "statistic":   "Average",
    },
    {
        "name":        f"{endpoint_name}-no-traffic",
        "description": "超過 30 分鐘沒有流量（可能 endpoint 掛了）",
        "metric":      "Invocations",
        "threshold":   0,
        "comparison":  "LessThanOrEqualToThreshold",
        "statistic":   "Sum",
        "period":      1800,   # 30 分鐘
    },
]

for alarm in alarms:
    cw.put_metric_alarm(
        AlarmName=alarm["name"],
        AlarmDescription=alarm["description"],
        Namespace="AWS/SageMaker",
        MetricName=alarm["metric"],
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName",  "Value": VARIANT},
        ],
        Period=alarm.get("period", 300),   # 5 分鐘
        EvaluationPeriods=2,
        Threshold=alarm["threshold"],
        ComparisonOperator=alarm["comparison"],
        Statistic=alarm.get("statistic", "Average"),
        AlarmActions=[topic_arn],          # 觸發時通知 SNS
        TreatMissingData="notBreaching",
    )
    print(f"  警報設定：{alarm['name']}")
    print(f"    → {alarm['description']}")

# ─────────────────────────────────────────
# 3. 建立 CloudWatch Dashboard
# ─────────────────────────────────────────
print("\n=== 3. 建立 CloudWatch Dashboard ===")

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "x": 0, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "推論延遲（毫秒）",
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency",
                     "EndpointName", endpoint_name,
                     "VariantName", VARIANT,
                     {"stat": "Average", "label": "P50"}],
                    ["...", {"stat": "p90", "label": "P90"}],
                    ["...", {"stat": "p99", "label": "P99"}],
                ],
                "period": 60,
                "view": "timeSeries",
                "yAxis": {"left": {"label": "微秒"}},
            },
        },
        {
            "type": "metric",
            "x": 12, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "每分鐘呼叫次數",
                "metrics": [
                    ["AWS/SageMaker", "Invocations",
                     "EndpointName", endpoint_name,
                     "VariantName", VARIANT,
                     {"stat": "Sum", "label": "Total Invocations"}],
                ],
                "period": 60,
                "view": "timeSeries",
            },
        },
        {
            "type": "metric",
            "x": 0, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "錯誤數",
                "metrics": [
                    ["AWS/SageMaker", "Invocation4XXErrors",
                     "EndpointName", endpoint_name,
                     "VariantName", VARIANT,
                     {"stat": "Sum", "label": "4XX 錯誤"}],
                    ["AWS/SageMaker", "Invocation5XXErrors",
                     "EndpointName", endpoint_name,
                     "VariantName", VARIANT,
                     {"stat": "Sum", "label": "5XX 錯誤"}],
                ],
                "period": 60,
                "view": "timeSeries",
            },
        },
        {
            "type": "alarm",
            "x": 12, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "警報狀態",
                "alarms": [
                    f"arn:aws:cloudwatch:{region}:{account_id}:alarm:{a['name']}"
                    for a in alarms
                ],
            },
        },
    ]
}

cw.put_dashboard(
    DashboardName="MLOps-SentimentEndpoint",
    DashboardBody=json.dumps(dashboard_body),
)

dashboard_url = (
    f"https://{region}.console.aws.amazon.com/cloudwatch/home"
    f"?region={region}#dashboards:name=MLOps-SentimentEndpoint"
)
print(f"Dashboard 建立完成：")
print(f"  {dashboard_url}")

# ─────────────────────────────────────────
# 4. 查看目前指標
# ─────────────────────────────────────────
print("\n=== 4. 目前 Endpoint 指標（最近 1 小時）===")

from datetime import datetime, timezone, timedelta

end_time   = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=1)

for metric_name in ["Invocations", "ModelLatency", "Invocation5XXErrors"]:
    try:
        result = cw.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName",  "Value": VARIANT},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=["Sum", "Average"],
        )
        datapoints = result.get("Datapoints", [])
        if datapoints:
            dp = datapoints[0]
            print(f"  {metric_name}: Sum={dp.get('Sum', 0):.0f}, Avg={dp.get('Average', 0):.2f}")
        else:
            print(f"  {metric_name}: 暫無資料（endpoint 可能剛啟動）")
    except Exception as e:
        print(f"  {metric_name}: 無法取得（{e}）")

print(f"""
=== 重點整理 ===

監控三個核心指標：
  ModelLatency     → 推論延遲，P99 > 2秒要警報
  Invocations      → 流量，突然歸零可能是 endpoint 掛了
  Invocation5XXErrors → 伺服器錯誤，> 5% 要警報

警報通知流程：
  CloudWatch 偵測到異常
    → 觸發 SNS Topic
    → 寄 Email 通知
    → 也可以接 Slack / PagerDuty

Dashboard URL：
  {dashboard_url}

進階：Model Drift 監控
  SageMaker Model Monitor 可以監控輸入資料分布
  當真實世界資料和訓練資料差太多時自動警報
  → 觸發重新訓練 pipeline
""")

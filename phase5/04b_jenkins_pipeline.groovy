// ============================================================
// 04b_jenkins_pipeline.groovy
// 主題：Jenkins Pipeline CI/CD
//
// 放在 repo 根目錄，命名為 Jenkinsfile
// Jenkins 偵測到 push 時自動執行。
//
// 流程：
//   push to main
//     → 跑測試
//     → 觸發 SageMaker Pipeline（訓練→評估→登記）
//     → 等待 pipeline 完成
//     → 若模型 Approved → 更新 Endpoint
// ============================================================

pipeline {
    agent {
        docker {
            // 使用 Python 3.11 容器執行
            image 'python:3.11'
            args '-u root'
        }
    }

    // Jenkins Credentials 設定
    // Jenkins → Manage Jenkins → Credentials → 加入以下 Secret Text：
    //   AWS_ACCESS_KEY_ID      → 你的 AWS Access Key
    //   AWS_SECRET_ACCESS_KEY  → 你的 AWS Secret Key
    //   AWS_DEFAULT_REGION     → us-west-2
    //   SAGEMAKER_ROLE_ARN     → SageMaker Execution Role ARN
    environment {
        AWS_ACCESS_KEY_ID     = credentials('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = credentials('AWS_SECRET_ACCESS_KEY')
        AWS_DEFAULT_REGION    = credentials('AWS_DEFAULT_REGION')
        SAGEMAKER_ROLE_ARN    = credentials('SAGEMAKER_ROLE_ARN')
    }

    // 觸發條件：只有 main branch 跑完整部署
    triggers {
        // 每天 AM 2:00 自動重新訓練（cron 格式）
        cron(env.BRANCH_NAME == 'main' ? 'H 2 * * *' : '')
    }

    stages {

        // ── Stage 1: 安裝套件 ──
        stage('Install Dependencies') {
            steps {
                sh '''
                    pip install pytest boto3 sagemaker torch transformers \
                                datasets evaluate --quiet
                '''
            }
        }

        // ── Stage 2: 跑測試 ──
        stage('Run Tests') {
            steps {
                sh 'pytest tests/ -v --tb=short --junitxml=test-results.xml || true'
            }
            post {
                always {
                    // 把測試結果顯示在 Jenkins UI
                    junit 'test-results.xml'
                }
            }
        }

        // ── Stage 3: 觸發 SageMaker Pipeline（只有 main branch）──
        stage('Trigger SageMaker Pipeline') {
            when {
                branch 'main'
            }
            steps {
                script {
                    def buildNumber = env.BUILD_NUMBER
                    sh """
                        python3 -c "
import boto3, os, time, json

region = os.environ['AWS_DEFAULT_REGION']
sm     = boto3.client('sagemaker', region_name=region)

# 觸發 SageMaker Pipeline
response = sm.start_pipeline_execution(
    PipelineName='mlops-sentiment-pipeline',
    PipelineExecutionDisplayName='jenkins-${buildNumber}',
    PipelineParameters=[
        {'Name': 'AccuracyThreshold', 'Value': '0.75'},
        {'Name': 'ModelApprovalStatus', 'Value': 'PendingManualApproval'},
    ],
)

execution_arn = response['PipelineExecutionArn']
print(f'Pipeline 啟動：{execution_arn}')

# 寫入 ARN 供下一個 stage 使用
with open('execution_arn.txt', 'w') as f:
    f.write(execution_arn)

# 等待完成
while True:
    status = sm.describe_pipeline_execution(
        PipelineExecutionArn=execution_arn
    )['PipelineExecutionStatus']
    print(f'狀態：{status}')
    if status in ['Succeeded', 'Failed', 'Stopped']:
        break
    time.sleep(30)

if status != 'Succeeded':
    raise Exception(f'Pipeline 失敗：{status}')

print('Pipeline 完成！')
                        "
                    """
                }
            }
        }

        // ── Stage 4: 部署（需要人工確認）──
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            // 部署前暫停等待人工確認（Jenkins input）
            input {
                message "確定要部署最新模型到 Production 嗎？"
                ok "Deploy"
                submitter "admin,ml-team"   // 只有這些人能確認
            }
            steps {
                sh '''
                    python3 -c "
import boto3, os, sagemaker
from sagemaker.huggingface import HuggingFaceModel

region = os.environ['AWS_DEFAULT_REGION']
role   = os.environ['SAGEMAKER_ROLE_ARN']
sm     = boto3.client('sagemaker', region_name=region)

# 取得最新 Approved 模型
packages = sm.list_model_packages(
    ModelPackageGroupName='mlops-sentiment-model-group',
    ModelApprovalStatus='Approved',
    SortBy='CreationTime',
    SortOrder='Descending',
)

if not packages['ModelPackageSummaryList']:
    print('沒有 Approved 的模型，跳過部署')
    exit(0)

latest_arn = packages['ModelPackageSummaryList'][0]['ModelPackageArn']
print(f'部署模型版本：{latest_arn}')

session = sagemaker.Session()
model   = HuggingFaceModel(
    model_data=latest_arn,
    role=role,
    transformers_version='4.37',
    pytorch_version='2.1',
    py_version='py310',
)
model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='mlops-sentiment-endpoint',
    update_endpoint=True,
)
print('Endpoint 更新完成！')
                    "
                '''
            }
        }
    }

    // ── Pipeline 結束後的動作 ──
    post {
        success {
            echo '✅ Pipeline 執行成功！'
            // 可以加 Slack 通知：slackSend channel: '#ml-deploy', message: '部署成功'
        }
        failure {
            echo '❌ Pipeline 執行失敗！'
            // 可以加 Email 通知：mail to: 'ml-team@company.com', subject: 'Pipeline 失敗'
        }
    }
}

// ============================================================
// Jenkins 設定說明：
//
// 1. 在 Jenkins 安裝 Plugin：
//    - Docker Pipeline
//    - AWS Credentials Plugin
//    - Pipeline
//
// 2. 建立 Pipeline Job：
//    Jenkins → New Item → Pipeline
//    → Pipeline script from SCM
//    → SCM: Git → 填入 Bitbucket repo URL
//    → Script Path: Jenkinsfile
//
// 3. Bitbucket Webhook：
//    Bitbucket → Repository → Settings → Webhooks
//    → Add webhook → URL: http://your-jenkins/github-webhook/
//    → Triggers: Push
//
// Bitbucket vs Jenkins 差異：
//   Bitbucket Pipelines → SaaS，設定簡單，在 Bitbucket 上直接跑
//   Jenkins             → 自架，彈性高，需要維護 Jenkins server
// ============================================================

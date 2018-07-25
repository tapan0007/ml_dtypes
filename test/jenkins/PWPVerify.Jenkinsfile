pipeline {
   agent {
        dockerfile {
            dir 'test/jenkins'
            customWorkspace './docker-shared'
            args '-u root -v /home/jenkins:/home/jenkins -v /home/jenkins/kaena/.aws:/root/.aws -v $WORKSPACE:/artifact'
            label 'kaena_c5.18xl'
        }
    }
    environment {
        SRC_DIR = "/workdir/src"

        KAENA_RT_PATH = "$SRC_DIR/krt"
        KRT_PWP_PATH = "$KAENA_RT_PATH/pwp"
    }
    stages {
        stage('Prep'){
            steps {
                sh 'cat /etc/os-release'
                sh 'cat /proc/cpuinfo'
                sh 'df -h'

                sh 'ls -ltrA'
                sh 'rm -rf $SRC_DIR && mkdir -p $SRC_DIR'
            }
        }
        stage('Checkout'){

            steps {
                sh '''
                set -x

                cd $SRC_DIR
                ls -ltrA

                repo init -u ssh://siopt.review/tonga/sw/kaena/manifest
                [ ! -z "$MANIFEST_FILE_NAME" ] || export MANIFEST_FILE_NAME=default.xml
                export MANIFEST_REPO_REFSPEC=$GERRIT_REFSPEC
                git clone  ssh://siopt.review/tonga/sw/kaena/manifest
                git -C manifest fetch origin $GERRIT_REFSPEC || export MANIFEST_REPO_REFSPEC=""
                [ -z "$MANIFEST_REPO_REFSPEC" ] || export MANIFEST_REPO_REFSPEC_OPT="-b $MANIFEST_REPO_REFSPEC"
                repo init -m $MANIFEST_FILE_NAME $MANIFEST_REPO_REFSPEC_OPT
                repo sync -j 8
                '''
            }
        }
        stage('Verify') {
            steps {
                sh 'export PYTHONUNBUFFERED=1 && cd $KRT_PWP_PATH && ./test_pwp.py'
            }
            post {
                always {
                    sh 'cp -r $KRT_PWP_PATH/pwp_cache /artifact/'
                    sh 'cp -r $KRT_PWP_PATH/log_act_config /artifact/'
                    sh 'cp -r $KRT_PWP_PATH/test_func_archive /artifact/'
                    archiveArtifacts artifacts:'pwp_cache/**/*'
                    archiveArtifacts artifacts:'log_act_config/**/*'
                    archiveArtifacts artifacts:'test_func_archive/**/*'
                }
            }
        }
    }
    post {
        always {
            sh 'df -h'
            cleanWs()
        }
    }
}

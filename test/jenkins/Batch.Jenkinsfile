
pipeline{

    agent {
        dockerfile {
            dir 'test/jenkins'
            customWorkspace './docker-shared'
            args '-u root -v /home/jenkins:/home/jenkins -v /home/jenkins/kaena/.aws:/root/.aws -v $WORKSPACE:/artifact'
            label "${env.AGENT_LABEL}"
        }
    }

    environment {
        AGENT_LABEL = "kaena_c5.18xl"
        SRC_DIR = "/workdir/src"
        BLD_DIR = "$SRC_DIR/build"
        TEST_DIR = "/workdir/test"

        KAENA_PATH = "$SRC_DIR/kcc"
        KAENA_EXT_PATH = "$SRC_DIR/ext"
        INKLING_PATH = "$SRC_DIR/inkling"
        QEMU_INKLING_PATH = "$SRC_DIR/qemu_inkling"
        ARCH_ISA_PATH = "$SRC_DIR/arch-isa"
        KAENA_RT_PATH = "$SRC_DIR/krt"
        ARCH_HEADERS_PATH = "$SRC_DIR/arch-headers"

        KRT_BLD_DIR = "$BLD_DIR/krt"
        KRT_DV_BLD_DIR = "$BLD_DIR/krt_dv"
        KRT_EMU_BLD_DIR = "$BLD_DIR/krt_emu"
        KCC_BLD_DIR = "$BLD_DIR/kcc"
        QEMU_BLD_DIR = "$BLD_DIR/qemu"
    }

    stages {
        stage('Prep'){
            steps {
                sh 'cat /etc/os-release'
                sh 'cat /proc/cpuinfo'
                sh 'df -h'

                sh 'ls -ltrA'
                sh 'rm -rf $SRC_DIR && mkdir -p $SRC_DIR'
                sh 'rm -rf $BLD_DIR && mkdir -p $BLD_DIR'
                sh 'mkdir -p $KRT_BLD_DIR $KRT_DV_BLD_DIR $QEMU_BLD_DIR $KCC_BLD_DIR'
                sh 'rm -rf $TEST_DIR && mkdir -p $TEST_DIR/hourly && mkdir -p $TEST_DIR/precheckin'
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

                git config --global user.name "Jenkins"
                git config --global user.email aws-tonga-kaena@amazon.com

                [ -z "$GERRIT_REFSPEC" ] || \
                git -C krt pull origin $GERRIT_REFSPEC || \
                git -C kcc pull origin $GERRIT_REFSPEC || \
                git -C ext pull origin $GERRIT_REFSPEC || \
                git -C inkling pull origin $GERRIT_REFSPEC || \
                git -C qemu_inkling pull origin $GERRIT_REFSPEC || \
                git -C arch-isa pull origin $GERRIT_REFSPEC || \
                git -C shared pull origin $GERRIT_REFSPEC || \
                git -C manifest pull origin $GERRIT_REFSPEC

                chmod -R 755 ./
                ls -ltrA
                '''
            }
        }
        stage('build') {
            stages {
                stage('krt') {
                    steps {
                        sh 'cd $KRT_BLD_DIR && cmake $KAENA_RT_PATH && make package'
                        sh 'cd $KRT_DV_BLD_DIR && PLAT=dv cmake $KAENA_RT_PATH && make package'
                    }
                } 
                stage('kaena') { 
                    steps {
                        sh 'cd $SRC_DIR/shared && ./build.sh'
                    }
                    post {
                        success {
                            sh 'cp $KRT_BLD_DIR/krt-*.*.tar.gz /artifact/'
                            sh 'cp $KRT_DV_BLD_DIR/krt-*.*.tar.gz /artifact/'
                            sh 'cp $KRT_EMU_BLD_DIR/krt-*.*.tar.gz /artifact/'
                            archiveArtifacts artifacts:'krt-*.*.tar.gz'
                        }
                    }
                }
            }
        }
        stage('Regressions') {
            stages {
                stage('batch') {
                    steps {
                        timeout(time: 24, unit: 'HOURS') {
                            sh 'cd $KAENA_PATH/test/tools && python3 batch_test_10k.py --aws-profile=kaena'
                        }
                    }
                    post {
                        always {
                            sh 'mkdir /artifact/batch'
                            sh '/bin/cp $KAENA_PATH/test/tools/*.csv /artifact/batch'
                            sh '/bin/cp $KAENA_PATH/test/tools/*.png /artifact/batch'
                            sh 'chmod -R a+wX /artifact/'
                            archiveArtifacts artifacts:'/artifact/batch/*'
                        }
                        failure {
                            sh 'mkdir /artifact/batch'
                            sh '/bin/cp $KAENA_PATH/test/tools/batch-fail.txt /artifact/batch'
                            sh 'chmod -R a+wX /artifact/'
                            archiveArtifacts artifacts:'/artifact/batch/*'
                        }
                    }
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



pipeline{

    agent {
        dockerfile {
            dir 'test/jenkins'
            customWorkspace './docker-shared'
            args '-u root -v /home/jenkins:/home/jenkins -v /home/jenkins/kaena/.aws:/root/.aws -v $WORKSPACE:/artifact -v /proj/trench/sw/kaena-test:/kaena-test'
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
        ARCH_ARTIFACTS_PATH = "$SRC_DIR/arch-headers"

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
                sh 'cat /proc/meminfo'
                sh 'df -h'
                sh 'ls -ltrA'


                sh 'rm -rf $SRC_DIR && mkdir -p $SRC_DIR'
                sh 'rm -rf $BLD_DIR && mkdir -p $BLD_DIR'
		sh 'mkdir -p /root/.ssh'
		sh 'cp /home/jenkins/.ssh/config /root/.ssh/config'
		sh 'cp /home/jenkins/.ssh/siopt-vpc.pem /root/.ssh/siopt-vpc.pem'
		sh 'cp /home/jenkins/.ssh/id_rsa /root/.ssh/id_rsa'
		sh 'chmod 600 /root/.ssh/siopt-vpc.pem'
		sh 'chmod 600 /root/.ssh/id_rsa'
		sh 'export $KAENA_ZEBU_SERVER
                sh 'rm -rf $TEST_DIR && mkdir -p $TEST_DIR/RunAllWithArgs && mkdir -p $TEST_DIR/precheckin && mkdir -p $TEST_DIR/RunPytest'
                sh '''
                [ -f "/kaena-test/ubuntu-18.04-24G_pytest.qcow2" ] && /bin/cp "/kaena-test/ubuntu-18.04-24G_pytest.qcow2" /tmp/ubuntu-18.04-24G_pytest.qcow2
                '''
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

                git -C krt  describe --always --dirty 
                git -C kcc  describe --always --dirty 
                git -C ext  describe --always --dirty 
                git -C inkling describe --always --dirty 
                git -C qemu_inkling describe --always --dirty 
                git -C arch-isa describe --always --dirty 
                git -C shared  describe --always --dirty 
                git -C manifest describe --always --dirty 

                cd $ARCH_HEADERS_PATH
                [ ! -z "$ARCH_HEADER_VERSION" ] || export ARCH_HEADER_VERSION=$(git describe --tags $(git rev-list --tags --max-count=1))
                git checkout $ARCH_HEADER_VERSION

                cd $ARCH_ISA_PATH
                [ ! -z "$ARCH_ISA_VERSION" ] || export $ARCH_ISA_VERSION=master
                git checkout $ARCH_ISA_VERSION


                chmod -R 755 ./
                ls -ltrA
                '''
            }
        }
        stage('build') {
            stages {
                stage('kaena') {
                    steps {
                        sh 'cd $SRC_DIR/shared && ./build.sh'
                        sh '[ -z "$SIM_DEBUG" ] || (cd $INKLING_PATH/sim && make clean && make opt)'
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
                stage('RunAllWithArgs') {
                    steps {
                        timeout(time: 5, unit: 'HOURS') {
                            sh '''
                            [ -z "$RUNALL_ARGS" ] || (cd $TEST_DIR/RunAllWithArgs && $KAENA_PATH/test/e2e/RunAll $RUNALL_ARGS)
                            '''
                        }
                    }
                    post {
                        always {
                            sh 'mkdir /artifact/RunAllWithArgs'
                            sh '/bin/cp $TEST_DIR/RunAllWithArgs/qor* /artifact/RunAllWithArgs/ || touch /artifact/RunAllWithArgs/qor_RunAllWithArgs_qor_available.txt'
                            sh 'chmod -R a+wX /artifact/'
                            archiveArtifacts artifacts:'RunAllWithArgs/qor*,*.tgz'
                        }
                        failure {
                            sh 'find $TEST_DIR/RunAllWithArgs -type f -name "*.vdi" -delete'
                            sh 'find $TEST_DIR/RunAllWithArgs -iname "*.txt" -print0 | tar -czvf /artifact/RunAllWithArgs/logs.tgz -T -'
                            sh 'chmod -R a+wX /artifact/'
                            archiveArtifacts artifacts:'RunAllWithArgs/logs.tgz'
                        }
                    }
                }
                stage('RunPytest') {
                    steps {
                        timeout(time: 50, unit: 'MINUTES') {
                            sh '''
                            [ -z "$RUNNC_ARGS" ] || (cd $TEST_DIR/RunPytest && $KAENA_PATH/runtime/util/qemu_rt $RUNNC_ARGS)
                            '''
                        }
                    }
                    post {
                        always {
			    sh '''
			    [ -z "$RUNNC_ARGS" ] || (/bin/cp $TEST_DIR/RunPytest/*.xml $WORKSPACE/.)
			    '''
			    junit allowEmptyResults: true, testResults: '*.xml'
                        }
                        failure {
                            sh 'find $TEST_DIR/RunPytest -type f -name "*.vdi" -delete'
                        }
                    }
                }
            }
        }
    }
    post {
        failure {
            script {
                 if (!env.GERRIT_REFSPEC) { 
                     // Send an email only if the build status has changed from green/unstable to red
                     emailext subject: '$DEFAULT_SUBJECT',
                     body: '$DEFAULT_CONTENT',
                     recipientProviders: [
                             [$class: 'CulpritsRecipientProvider'],
                             [$class: 'DevelopersRecipientProvider'],
                             [$class: 'RequesterRecipientProvider']
                     ],
                       replyTo: '$DEFAULT_REPLYTO',
                     to: 'aws-tonga-kaena@amazon.com'
                 }
            }
        }
        always {
            sh 'df -h'
            sh 'cat /proc/meminfo'
            sh 'ls -ltrA'
            cleanWs()
        }
    }
}


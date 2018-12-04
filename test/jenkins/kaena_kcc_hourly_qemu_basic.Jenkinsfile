
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
                sh 'rm -rf $TEST_DIR'
                sh 'mkdir -p $TEST_DIR/test_qemu_compiler'
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


                chmod -R 755 ./
                ls -ltrA
                '''
            }
        }
        stage('build') {
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
        stage('test_qemu') {
            stages {
                stage('compiler_test') {
                    steps {
                        timeout(time: 12, unit: 'HOURS') {
                            sh '''
                            (cd $TEST_DIR/test_qemu_compiler  && $KAENA_PATH/test/e2e/RunAll --level 10 --force_qemu --filter not_waived --max_load 100)
                            '''
                        }
                    }
                    post {
                        always {
                            sh '''
                            ([ -f $TEST_DIR/test_qemu_compiler/RunAllReport.xml ] && /bin/cp $TEST_DIR/test_qemu_compiler/RunAllReport.xml $WORKSPACE/RunAllReportFull.xml)
                            '''
                            junit allowEmptyResults: true, testResults: 'RunAllReportFull.xml'
                            sh 'mkdir /artifact/test_qemu_compiler'
                            sh '/bin/cp $TEST_DIR/test_qemu_compiler/qor* /artifact/test_qemu_compiler/ || touch /artifact/test_qemu_compiler/qor_RunAllWithArgs_qor_available.txt'
                            sh 'for f in `find $TEST_DIR/test_qemu_compiler  -iname "*.txt" -o -iname "*.json" -o -iname "*.bin" -o -iname "*.svg" -o -iname "*.png" -o -iname "*.csv" -o -iname "*.asm" `; do cp $f --parents  /artifact/test_qemu_compiler/;done; '
                            sh 'for f in `find $TEST_DIR/test_qemu_compiler/*/working_dir  -iname "*.npy" -a -type f`; do cp $f --parents  /artifact/test_qemu_compiler/;done; '
                            sh 'chmod -R a+wX /artifact/'
                            archiveArtifacts artifacts:'test_qemu_compiler/*.txt,*.tgz,test_qemu_compiler/**/*.txt,tgz,test_qemu_compiler/**/*.json, test_qemu_compiler/**/*.bin, test_qemu_compiler/**/*.svg, test_qemu_compiler/**/*.png, test_qemu_compiler/**/*.csv, test_qemu_compiler/**/*.asm, test_qemu_compiler/**/*.npy'
                        }
                        failure {
                            sh 'find $TEST_DIR/test_qemu_compiler -type f -name "*.vdi" -delete'
                            sh 'find $TEST_DIR/test_qemu_compiler -iname "*.txt" -print0 | tar -czvf /artifact/test_qemu_compiler/logs.tgz -T -'
                            sh 'chmod -R a+wX /artifact/'
                            archiveArtifacts artifacts:'test_qemu_compiler/logs.tgz'
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
                     body: '''${SCRIPT, template="groovy-html-change.template"}''',
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


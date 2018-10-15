pipeline {

   agent {
        dockerfile {
            dir 'test/jenkins'
            customWorkspace './docker-shared'
            args '-u root -v /home/jenkins:/home/jenkins -v /home/jenkins/kaena/.aws:/root/.aws -v $WORKSPACE:/artifact'
            label "${env.AGENT_LABEL}"
        }
    }

    environment {
        AGENT_LABEL          = "kaena_c5.18xl"
        AWS_TONGA_SRC        = "/workdir/src"

        TEST_DIR             = "/workdir/test"
        SRC_DIR              = "/workdir/src"
        BLD_DIR              = "$SRC_DIR/build"

        ARCH_ARTIFACTS_PATH  =  "$AWS_TONGA_SRC/arch-headers"
        ARCH_HEADERS_PATH    =  "$AWS_TONGA_SRC/arch-headers "
        ARCH_ISA_PATH        =  "$AWS_TONGA_SRC/arch-isa"
        INKLING_INSTALL_DIR  =  "$AWS_TONGA_SRC/build/tmp_install_inkling"
        INKLING_PATH         =  "$AWS_TONGA_SRC/inkling"
        KAENA_EXT_PATH       =  "$AWS_TONGA_SRC/ext"
        KAENA_PATH           =  "$AWS_TONGA_SRC/kcc"
        KAENA_RT_PATH        =  "$AWS_TONGA_SRC/krt"
        KCC_INSTALL_DIR      =  "$AWS_TONGA_SRC/build/tmp_install_kcc"
        QEMU_INKLING_PATH    =  "$AWS_TONGA_SRC/qemu_inkling"
        TONGA_RTL            =  "$AWS_TONGA_SRC/rtl"

        KCC_BLD_DIR          =  "$BLD_DIR/kcc"
        KRT_BLD_DIR          =  "$BLD_DIR/krt"
        KRT_DV_BLD_DIR       =  "$BLD_DIR/krt_dv"
        KRT_EMU_BLD_DIR      =  "$BLD_DIR/krt_emu"
        QEMU_BLD_DIR         =  "$BLD_DIR/qemu"
        QEMU_INSTALL_DIR     =  "$BLD_DIR/tmp_install_qemu"
        PYTHONPATH           =  "$AWS_TONGA_SRC/tvm/python:$AWS_TONGA_SRC/tvm/nnvm/python:$AWS_TONGA_SRC/tvm/topi/python"
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

                repo init -u ssh://siopt.review/tonga/sw/kaena/manifest -m master.xml
                [ ! -z "$MANIFEST_FILE_NAME" ] || export MANIFEST_FILE_NAME=default.xml
                export MANIFEST_REPO_REFSPEC=$GERRIT_REFSPEC
                git clone  ssh://siopt.review/tonga/sw/kaena/manifest
                git -C manifest fetch origin $GERRIT_REFSPEC || export MANIFEST_REPO_REFSPEC=""
                [ -z "$MANIFEST_REPO_REFSPEC" ] || export MANIFEST_REPO_REFSPEC_OPT="-b $MANIFEST_REPO_REFSPEC"
                repo init -m $MANIFEST_FILE_NAME $MANIFEST_REPO_REFSPEC_OPT
                repo sync -j 8

                # Output the base revisions we have in all repos after repo sync
                repo info
                repo forall -p -c git rev-parse HEAD

                git config --global user.name "Jenkins"
                git config --global user.email aws-tonga-kaena@amazon.com

                [ -z "$GERRIT_REFSPEC" ] || \
                git -C starfish pull origin $GERRIT_REFSPEC || \
                git -C tvm pull origin $GERRIT_REFSPEC || \
                git -C tvm/dmlc-core pull origin $GERRIT_REFSPEC || \
                git -C tvm/HalideIR pull origin $GERRIT_REFSPEC || \
                git -C tvm/dlpack pull origin $GERRIT_REFSPEC || \
                git -C kcc pull origin $GERRIT_REFSPEC # kcc repo needed for the sealife_jenkinsfile builder

                chmod -R 755 ./
                '''
            }
        }

        stage('Build') {
            steps {
                sh 'mkdir -p $BLD_DIR && cd $BLD_DIR && $AWS_TONGA_SRC/shared/build_sealife.sh'
            }
        }

        stage('Verify') {
            steps {
                // TODO: sh 'mkdir -p $TEST_DIR && cd $AWS_TONGA_SRC/tvm && ./tests/scripts/task_python_topi.sh 2>&1 | tee $TEST_DIR/log-tvm.txt'
                sh 'mkdir -p $TEST_DIR/starfish && cd $TEST_DIR/starfish && $AWS_TONGA_SRC/starfish/test/run.sh'
           }
            post {
                always {
                    sh 'mkdir /artifact/starfish'
                    sh 'for f in `find $TEST_DIR/starfish -iname "*.txt" -o -iname "*.json" -o -iname "*.bin" -o -iname "*.svg" `; do cp $f --parents  /artifact/starfish/; done; '
                    archiveArtifacts artifacts:'starfish/**'
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
            cleanWs()
        }
    }
}

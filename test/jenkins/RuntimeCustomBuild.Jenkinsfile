
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
                sh 'rm -rf $SRC_DIR && mkdir -p $SRC_DIR'
                sh 'rm -rf $BLD_DIR && mkdir -p $BLD_DIR'
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

                cd $ARCH_HEADERS_PATH
                [ ! -z "$ARCH_HEADER_VERSION" ] || export ARCH_HEADER_VERSION=$(git describe --tags $(git rev-list --tags --max-count=1))
                git checkout $ARCH_HEADER_VERSION

                cd $ARCH_ISA_PATH
                [ ! -z "$ARCH_ISA_VERSION" ] || export $ARCH_ISA_VERSION=master
                git checkout $ARCH_ISA_VERSION

                cd $KAENA_RT_PATH
                [ ! -z "$KRT_VERSION" ] || export $KRT_VERSION=master
                git checkout $KRT_VERSION

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
                    }
                    post {
                        success {
                            sh 'cp $KRT_BLD_DIR/*.*.tar.gz /artifact/'
                            sh 'cp $KRT_DV_BLD_DIR/*.*.tar.gz /artifact/'
                            sh 'cp $KRT_EMU_BLD_DIR/*.*.tar.gz /artifact/'
                            archiveArtifacts artifacts:'krt-*.*.tar.gz'
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}


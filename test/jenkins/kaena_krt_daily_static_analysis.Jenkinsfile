pipeline{
    agent {
        dockerfile {
            file 'StaticAnalysis.Dockerfile'
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
    }

    stages {
        stage('Prep'){
            steps {
                sh 'rm -rf $SRC_DIR && mkdir -p $SRC_DIR'
                sh 'rm -rf $BLD_DIR && mkdir -p $BLD_DIR'
                sh 'rm -rf $TEST_DIR
            }
        }
        stage('Checkout'){

            steps {
                sh '''
                set -x

                cd $SRC_DIR

                repo init -u ssh://siopt.review/tonga/sw/kaena/manifest
                repo sync -j 8

                '''
            }
        }
        stage('build') {
            stages {
                stage('scan-build') {
                    steps {
                        sh '''
                        cd $TEST_DIR
                        mkdir stats-bld && cd stats-bld
                        scan-build-6.0 -stats -o ./static_analysis cmake $KAENA_RT_PATH
                        scan-build-6.0 -stats -o ./static_analysis make -j 8
                        '''
                    }
                    post {
                        always {
                            sh 'cp $TEST_DIR/*.* /artifact/'
                            archiveArtifacts artifacts:'*.*'
                        }
                    }
                }
            }
        }
    }
}


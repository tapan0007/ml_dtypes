pipeline {
    agent { label 'tonga-docker' }
    environment {
        SRC_DIR = "src"
        KAENA_RT_PATH = "$SRC_DIR/krt"

        BLD_DIR = "build"
        KRT_DV_BLD_DIR = "$BLD_DIR/krt_dv"
    }
    stages {
        stage('Checkout'){
            agent {
                label 'tonga-nodocker'
            }
            steps {
                sh '''
                set -x

                rm -rf $SRC_DIR && mkdir -p $SRC_DIR && cd $SRC_DIR
                git clone ssh://siopt.review/tonga/sw/kaena/krt
                git clone ssh://siopt.review/tonga/sw/kaena/kcc
                git clone ssh://siopt.review/tonga/sw/arch-artifacts arch-headers
                cd arch-headers
                git checkout tonga-arch-artifacts-v0.2
                cd ..
                git clone ssh://siopt.review/tonga/arch-isa isa
                '''

                stash name: 'src_dir', includes: 'src/**/*'
            }
        }
        stage('Build') {
            agent {
                dockerfile {
                    dir 'test/jenkins'
                    customWorkspace './docker-shared'
                    args '-u root -v /home/jenkins:/home/jenkins -v /home/jenkins/kaena/.aws:/root/.aws -v $WORKSPACE:/artifact'
                    label 'kaena_c5.18xl'
                }
            }
            steps {
                sh 'rm -rf $BLD_DIR && mkdir -p $BLD_DIR'
                unstash 'src_dir'
                sh 'ls src'
                sh 'mkdir -p $KRT_DV_BLD_DIR'

                sh 'cd $KRT_DV_BLD_DIR && PLAT=dv cmake ../../$KAENA_RT_PATH && make package'
            }
            post {
                success {
                    sh 'cp $KRT_DV_BLD_DIR/krt-*.*-dv-hal.tar.gz /artifact/'
                    archiveArtifacts artifacts:'krt-*.*-dv-hal.tar.gz'
                    sh 'chown 506:505 /artifact/*.*'
                    stash includes: 'krt-*.*-dv-hal.tar.gz', name: 'krt_dv_package'
                }
            }
        }
        stage('Release') {
            agent {
                label 'tonga-nodocker'
            }

            steps {
                script {
                    unstash 'krt_dv_package'
                    sh 'test/jenkins/dv_release.sh'
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


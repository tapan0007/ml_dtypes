pipeline {
    agent { label 'tonga-docker' }
    environment {
        SRC_DIR = "src"
        KAENA_RT_PATH = "$SRC_DIR/krt"

        BLD_DIR = "build"
        KRT_DV_BLD_DIR = "$BLD_DIR/krt_dv"
    }
    stages {
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
                sh '''
                set -x

                rm -rf $SRC_DIR && mkdir -p $SRC_DIR && cd $SRC_DIR
                repo init -u ssh://siopt.review/tonga/sw/kaena/manifest
                repo sync -j 8 krt arch-isa arch-headers ext
                rm -rf ext/apps ext/images
                cd ..
                '''

                sh 'rm -rf $BLD_DIR && mkdir -p $BLD_DIR'
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
                    sh 'ls -l'
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


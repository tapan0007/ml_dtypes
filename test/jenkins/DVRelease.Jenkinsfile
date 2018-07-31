pipeline{
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
        BLD_DIR = "/workdir/build"
        TEST_DIR = "/workdir/test"

        QEMU_INKLING_PATH = "$SRC_DIR/qemu_inkling"
        ARCH_ISA_PATH = "$SRC_DIR/arch-isa"
        ARCH_HEADERS_PATH = "$SRC_DIR/arch-headers"
        KAENA_RT_PATH = "$SRC_DIR/krt"

        KRT_BLD_DIR = "$BLD_DIR/krt"
        KRT_DV_BLD_DIR = "$BLD_DIR/krt_dv"
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
                sh 'rm -rf $TEST_DIR && mkdir -p $TEST_DIR'
            }
        }
        stage('Checkout'){

            steps {
                sh '''
                set -x

                cd $SRC_DIR

                repo init -u ssh://siopt.review/tonga/sw/kaena/manifest
                [ -z "$MANIFEST_FILE_NAME" ] && export MANIFEST_FILE_NAME=default.xml
                repo init -m $MANIFEST_FILE_NAME
                repo sync -j 8

                '''
            }
        }
        stage('Build') {
            steps {
                sh 'cd $KRT_DV_BLD_DIR && PLAT=dv cmake $KAENA_RT_PATH && make package'
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
                dir('/proj/trench/sw/kaena-release') {
                    sh 'rm -f *.tar.gz'
                    unstash 'krt_dv_package'
                    sh 'basename *.tar.gz tar.gz | xargs -I{} mkdir {}'
                    sh 'rm -f latest'
                    sh 'basename *.tar.gz tar.gz | xargs -I{} ln -s {} latest'
                    sh 'tar xvf *.tar.gz -C latest'
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


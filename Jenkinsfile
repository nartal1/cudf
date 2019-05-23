pipeline {

    agent {
        docker {
            label 'docker'
            image 'gpuci/rapidsai-base:cuda10.0-ubuntu18.04-gcc7-py3.7'
        }
    }

    options {
        ansiColor('xterm')
        timestamps()
        timeout(time: 1, unit: 'HOURS')
    }

    parameters {
        string(name: 'REF',
                defaultValue: '\${gitlabBranch}',
                description: 'Commit to build')
    }

    stages {
        stage('Compile') {
            steps {
                sh 'ci/cpu/build.sh' 
                sh 'ci/gpu/build.sh'
            }
        }
    }
}

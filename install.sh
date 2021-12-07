
root_dir=${1:-/content/setup/}
mkdir -p $root_dir
cd $root_dir

# download miniconda3
wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $root_dir/miniconda3 -f

# Set channels
#$root_dir/miniconda3/bin/conda config --add channels defaults
#$root_dir/miniconda3/bin/conda config --add channels bioconda
#$root_dir/miniconda3/bin/conda config --add channels conda-forge

# create py38 env
$root_dir/miniconda3/bin/conda create --name antispoof python=3.6.10 -y 
source $root_dir/miniconda3/bin/activate antispoof

# install other libraries for building TextFuseNet-detectron2. You can get detailed versions from the requirements.txt, 
# and difference between different versions may lead to unknown influence on performance.
pip install tensorflow==1.12.3

#
from setuptools import setup,find_packages

def install_other_pakage():
    with open('C:/Users/LENOVO/Desktop/Phishing/Phishing/requirements.txt', 'r') as f:
        lines = f.readlines()
        packages = [i.replace('/n','') for i in lines if i != '-e .' and not i.startswith('#')]
        return packages
    
setup(
    name='Phishing Website Detection',
    version='0.1.dev0',
    description="A simple phishing website detection tool",
    author='Mandalor-09',
    author_email='oms421621@gmail.com',
    packages=find_packages(),
    install_requires = install_other_pakage()
)
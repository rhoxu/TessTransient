from setuptools import find_packages, setup

NAME = 'TessTransient'
DESCRIPTION = 'Transient event detector for TESS'
URL = 'https://github.com/rhoxu/TessTransient'
EMAIL = 'hugh11rox@gmail.com'
AUTHOR ='Hugh Roxburgh'
VERSION = '1.0.0'
REQUIRED = ['lightkurve>=2.0.0',
            'tessreduce',
            'astrocut',
            'ffmpeg'
            'numpy',           
            'photutils>=1.4',
            'scipy',
            'astropy',
            'joblib',
            'multiprocess']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author_email=EMAIL,
    author=AUTHOR,
    license='MIT',
    packages=['TessTransient'],
)


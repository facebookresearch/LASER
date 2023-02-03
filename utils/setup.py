from setuptools import find_packages, setup

setup(
    name="sentence_cleaner_splitter",
    version="1.0.1",
    url="https://github.com/facebookresearch/LASER/",
    author="NLLB Data Team",
    author_email="nllb_data@fb.com",
    description="Clean and split sentences",
    packages=["sentence_cleaner_splitter"],
    package_dir={"sentence_cleaner_splitter": "src"},
    install_requires=[
        "indic-nlp-library==0.81",
        "sentence-splitter==1.4",
        "botok==0.8.8",
        "khmer-nltk==1.5",
        "LaoNLP==0.6",
        "sacremoses==0.0.43",
        "xxhash==3.0.0",
        "emoji==1.7.0",
    ],
)

from setuptools import setup

setup(
    author='Pavol Harar',
    author_email='pavol.harar@gmail.com',
    name='redistributor',
    version='1.0.0',
    license='MIT',
    py_modules=['redistributor'],
    description='''
    A tool for automatic transformation of empirical data distributions''',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    keywords='distribution transformation',
    project_urls={
        'Source': 'https://gitlab.com/paloha/redistributor',
        'Documentation': 'https://gitlab.com/paloha/redistributor/-/raw/master/docs/redistributor.pdf?inline=false',
        'Publication DOI': 'todo add doi',
        'Issue tracker': 'https://gitlab.com/paloha/redistributor/issues',
    },
    python_requires='>3.6',
    platforms=['Ubuntu'],
    install_requires=['numpy>=1.14.5', 'scipy>=1.1.0',
                      'scikit_learn>=0.19.1', 'psutil>=5.4.6'],
    extras_require={
        'dev': ['pdoc3',  # Documentation
                'pyflakes',  # Codecheck
                'jupyter',  # Examples
                ],
    }
)

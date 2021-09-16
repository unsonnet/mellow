from setuptools import setup, find_packages

# This sets up the mellow package
setup(
    name="mellow",
    version="0.0.1",
    description="A Deep NN model optimization algorithm.",
    author="Santiago Rodriguez",
    author_email="srodvasquez@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "jax", "jaxlib", "prox_tv"],
    url="https://github.com/QubicLens/mellow",
    license="Apache-2.0",
)

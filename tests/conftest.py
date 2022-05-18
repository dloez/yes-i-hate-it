"""Configuration for pytest tests"""
import shutil
import pytest

from yes_i_hate_it.main import OLD_TWEET_PATH


@pytest.fixture(autouse=True)
def check_data_dir(request):
    """Delete data directory"""
    def remove_data():
        if OLD_TWEET_PATH.exists():
            shutil.rmtree(OLD_TWEET_PATH.parent)
    request.addfinalizer(remove_data)

import os
from contextlib import closing

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from book_processor.Book import PROJ_ROOT
from pathlib import Path

UBLOCK_PATH = PROJ_ROOT.joinpath("1.17.0_0")
UBLOCK = Options()
UBLOCK.add_argument('load-extension=' + str(UBLOCK_PATH))
UBLOCK.add_argument('--ignore-certificate-errors')
UBLOCK.add_argument('--ignore-ssl-errors')

BROWSER_PATH = r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'

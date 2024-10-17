import streamlit.web.cli as stcli
import os
import sys

def streamlit_run():
    src = os.path.dirname(sys.executable) + '/webui.py'
    sys.argv=['streamlit', 'run', src, '--global.developmentMode=false']
    sys.exit(stcli.main())

if __name__ == "__main__":
    streamlit_run()
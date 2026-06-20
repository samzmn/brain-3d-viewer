import sys

from PyQt5 import QtWidgets

import base_viewer
import dbs_labeling
import abscess_labeling

def main(viewer_type: str = "dbs"):
    app = QtWidgets.QApplication(sys.argv)
    if viewer_type == "base":
        viewer = base_viewer.ViewerApp()
    elif viewer_type == "dbs":
        viewer = dbs_labeling.ViewerApp()
    elif viewer_type == "abscess":
        viewer = abscess_labeling.ViewerApp()
    else:
        raise ValueError("Invalid viewer type. Choose 'dbs' or 'abscess'.")
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main("base")

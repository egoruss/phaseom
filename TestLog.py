# !python.exe
# coding: cp1251
#

from __future__ import with_statement
import os
from datetime import datetime, date, time
from time import *
from types import *
import sip
import logging

sip.setapi('QVariant', 2)
from PyQt4 import QtCore, QtGui

newDir         = 'c:\\Soft\\PhaseOM\\trunk'
dataDir        = newDir + '\\EnergyEfficiency'
dataDirXls     = newDir + '\\EnergyEfficiency'
totalDir       = newDir + '\\Total'


class Logger(object):
    def __init__(self, output, lCursor):
        self.output = output
        self.lC = lCursor
        logging.basicConfig(format = u'[%(asctime)s] %(message)s', level = logging.DEBUG, filename = u'PhaseOM.log')

    def write(self, string):
        if not (string == "\n"):
            trstring = QtGui.QApplication.translate("MainWindow", string.replace("\n", '').decode('cp1251').strip(),
                                                    None, QtGui.QApplication.UnicodeUTF8)
            self.output.append(trstring)
            logging.info( trstring )
            self.output.moveCursor(self.lC.End, mode=self.lC.MoveAnchor)


class myQThread(QtCore.QThread):
    def __init__(self, output):
        QtCore.QThread.__init__(self)
        self.output = output

    def run(self):
        pass
        t0 = time()
        print ("Start : (���������� ������): %s" % ctime(t0))

        import MainPhaseOM

        t01 = time()
        t1 = time() - t0
        print ("Finis : ������ �� ������ (���������� ������): %.2f ���." % (t1))


class ProgressBar(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.pbar = QtGui.QProgressBar()


class MainWindow(QtGui.QMainWindow):
    def __init__(self, fileName=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(u'���������� ����������������� ������')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.isUntitled = True
        #        self.main_widget = QtGui.QWidget()
        #        self.setCentralWidget(self.main_widget)

        self.progress = QtGui.QProgressBar(self)
        self.progress.setMaximum(100)
        self.downDock = QtGui.QDockWidget()
        self.downDock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.downDock.setWidget(self.progress);
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.downDock)

        #        self.vbox = QtGui.QVBoxLayout(self)
        #        self.vbox.addWidget(self.logText)
        #        self.vbox.addWidget(self.progress)
        #        self.main_widget.setLayout(self.vbox)
        #        self.setLayout(self.vbox)

        self.logText = QtGui.QTextEdit()
        self.setCentralWidget(self.logText)
        # ���� ������� �������� read-only
        self.logText.setReadOnly(True)
        self.logText.setCurrentFont(QtGui.QFont('Courier New', 9))
        # ������ ������ ������������ ��������� ������� ������
        self.logText.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.logCursor = QtGui.QTextCursor()
        # ����� ��� �������������� ��� � ���� ������, ��� � ��������� ������ � 
        # ����������� �����
        self.logger = Logger(self.logText, self.logCursor)
        self.errors = Logger(self.logText, self.logCursor)
        sys.stdout = self.logger
        #        sys.stderr = self.errors
        #        sys.stderr = self.logger

        self.readSettings()
        self.createStatusBar()
        print '�������� ���� ���������.'
        # ��������� ���������
        self.logText.append(".:: ����� ���� ::.".decode('cp1251'))
        self.progress.setValue(1)

    @QtCore.pyqtSignature("")
    def threadFinished(self):
        self.logger.write("����� ��������!")

    def closeEvent(self, event):
        self.writeSettings()
        event.accept()

    def createStatusBar(self):
        self.statusBar().showMessage(u"������")

    def readSettings(self):
        print os.getcwd()
        settings = QtCore.QSettings('c:\\Soft\\PhaseOM\\trunk' + '\\phaseom.ini', 1)
        #        settings = QtCore.QSettings('StepService', 'RowsRatio')
        #        settings.setPath( )
        pos = settings.value('pos', QtCore.QPoint(200, 200))
        size = settings.value('size', QtCore.QSize(400, 400))
        self.move(pos)
        self.resize(size)

    def writeSettings(self):
        settings = QtCore.QSettings('c:\\Soft\\PhaseOM\\trunk' + '\\phaseom.ini', 1)
        #        settings = QtCore.QSettings('StepService', 'RowsRatio')
        settings.setValue('pos', self.pos())
        settings.setValue('size', self.size())


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    #    app.qRegisterMetaType(QtGui.QTextCursor())
    mainWin = MainWindow()
    mainWin.show()
    logTread = myQThread(mainWin.logger)
    print '������� ������ ������ ������ � ����.'
    logTread.start()

    sys.exit(app.exec_())

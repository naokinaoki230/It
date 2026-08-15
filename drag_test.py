import sys
import os

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QSplitter
)

from PyQt6.QtCore import (
    Qt,
    QMimeData,
    QUrl
)

from PyQt6.QtGui import QDrag

from PyQt6.QtWebEngineWidgets import QWebEngineView

from PyQt6.QtWebEngineCore import (
    QWebEngineProfile,
    QWebEnginePage,
    QWebEngineSettings
)


class DraggableListWidget(QListWidget):
    """ファイルをドラッグ可能なリスト"""

    def mouseMoveEvent(self, event):

        item = self.currentItem()

        if item and event.buttons() == Qt.MouseButton.LeftButton:

            file_path = item.data(Qt.ItemDataRole.UserRole)

            drag = QDrag(self)

            mime_data = QMimeData()

            mime_data.setUrls([
                QUrl.fromLocalFile(file_path)
            ])

            drag.setMimeData(mime_data)

            print(f"Drag Start : {file_path}")

            drag.exec(Qt.DropAction.CopyAction)

        super().mouseMoveEvent(event)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ドラッグ＆ドロップ テスト")
        self.resize(1400, 800)

        self.setup_ui()

    def setup_ui(self):

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -----------------------------
        # 左エリア
        # -----------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        btn_select = QPushButton("ファイル選択")
        btn_select.clicked.connect(self.select_files)

        self.file_list = DraggableListWidget()

        left_layout.addWidget(btn_select)
        left_layout.addWidget(self.file_list)

        left_widget.setLayout(left_layout)

        # -----------------------------
        # 右エリア（ブラウザ）
        # -----------------------------
        self.browser = QWebEngineView()

        #
        # プロファイル永続化設定
        #
        profile = QWebEngineProfile(
            "drag_test_profile",
            self.browser
        )

        storage_path = os.path.join(
            os.path.expanduser("~"),
            ".DragTestBrowser",
            "storage"
        )

        cache_path = os.path.join(
            os.path.expanduser("~"),
            ".DragTestBrowser",
            "cache"
        )

        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(cache_path, exist_ok=True)

        profile.setPersistentStoragePath(storage_path)
        profile.setCachePath(cache_path)

        profile.setHttpCacheType(
            QWebEngineProfile.HttpCacheType.DiskHttpCache
        )

        settings = profile.settings()

        settings.setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptEnabled,
            True
        )

        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalStorageEnabled,
            True
        )

        settings.setAttribute(
            QWebEngineSettings.WebAttribute.AutoLoadImages,
            True
        )

        settings.setAttribute(
            QWebEngineSettings.WebAttribute.DnsPrefetchEnabled,
            True
        )

        page = QWebEnginePage(
            profile,
            self.browser
        )

        self.browser.setPage(page)

        #
        # テストサイト
        #
        self.browser.setUrl(
            QUrl("https://chatgpt.com/c/6a7aea78-6378-83e8-a99f-6dd3d8a19342")
        )

        print("================================")
        print("ブラウザプロファイル初期化")
        print(f"Storage : {storage_path}")
        print(f"Cache   : {cache_path}")
        print("Cookie / LocalStorage が保存されます")
        print("================================")

        splitter.addWidget(left_widget)
        splitter.addWidget(self.browser)

        splitter.setSizes([350, 1050])

        self.setCentralWidget(splitter)

    def select_files(self):

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "ファイル選択"
        )

        for file_path in files:

            filename = os.path.basename(file_path)

            item = QListWidgetItem(filename)

            item.setData(
                Qt.ItemDataRole.UserRole,
                file_path
            )

            self.file_list.addItem(item)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
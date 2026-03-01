import sys
import threading
import json
from datetime import datetime, timezone

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeView, QTextEdit, QPushButton, QLabel, QSplitter, QMenu,
    QInputDialog, QMessageBox, QFrame, QHeaderView, QStyleFactory, QDialog, QCheckBox, QDialogButtonBox
)
from PyQt6.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QFont,
    QTextCursor, QTextCharFormat, QBrush, QPalette
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QModelIndex, QMimeData

import database as db
from main import TranscriberEngine, AppConfig, default_output_paths, UtteranceRecord, ensure_ollama

SPEAKER_COLORS_DARK = [
    "#60a5fa", "#34d399", "#a78bfa", "#fbbf24",
    "#f472b6", "#2dd4bf", "#fb7185", "#818cf8",
]
SPEAKER_COLORS_LIGHT = [
    "#2563eb", "#059669", "#7c3aed", "#d97706",
    "#db2777", "#0d9488", "#e11d48", "#4f46e5",
]


def _is_dark_mode(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def _build_stylesheet(dark: bool) -> str:
    if dark:
        return """
            QMainWindow { background-color: #0f172a; }
            QWidget#sidebar { background-color: #1e293b; }
            QTreeView {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                font-size: 14px;
                outline: none;
            }
            QTreeView::item:selected { background-color: #334155; }
            QTreeView::item:hover { background-color: #293548; }
            QTextEdit {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 10px;
            }
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4f46e5; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
            QLabel { color: #e2e8f0; }
            QSplitter::handle { background-color: #334155; width: 1px; }
            QMenu { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
            QMenu::item:selected { background-color: #334155; }
        """
    return """
        QMainWindow { background-color: #f8fafc; }
        QWidget#sidebar { background-color: #f1f5f9; }
        QTreeView {
            background-color: #ffffff;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            outline: none;
        }
        QTreeView::item:selected { background-color: #e0e7ff; }
        QTreeView::item:hover { background-color: #f1f5f9; }
        QTextEdit {
            background-color: #ffffff;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 10px;
        }
        QPushButton {
            background-color: #6366f1;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #4f46e5; }
        QPushButton:disabled { background-color: #e2e8f0; color: #94a3b8; }
        QLabel { color: #1e293b; }
        QSplitter::handle { background-color: #e2e8f0; width: 1px; }
        QMenu { background-color: #ffffff; color: #1e293b; border: 1px solid #e2e8f0; }
        QMenu::item:selected { background-color: #e0e7ff; }
    """

MIME_SESSION_ID = "application/x-intent-session-id"


class SessionTreeModel(QStandardItemModel):
    """Custom model that only allows dragging sessions onto folders."""

    session_moved = pyqtSignal(str, object)  # session_id, folder_id (str or None)

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        return [MIME_SESSION_ID]

    def mimeData(self, indexes):
        mime = QMimeData()
        for index in indexes:
            item = self.itemFromIndex(index)
            if item and item.data(Qt.ItemDataRole.UserRole + 1) == "session":
                mime.setData(MIME_SESSION_ID, item.data(Qt.ItemDataRole.UserRole).encode())
                return mime
        return mime

    def canDropMimeData(self, data, action, row, column, parent):
        if not data.hasFormat(MIME_SESSION_ID):
            return False
        target = self.itemFromIndex(parent)
        if target is None:
            return False
        return target.data(Qt.ItemDataRole.UserRole + 1) == "folder"

    def dropMimeData(self, data, action, row, column, parent):
        if not data.hasFormat(MIME_SESSION_ID):
            return False
        target = self.itemFromIndex(parent)
        if target is None or target.data(Qt.ItemDataRole.UserRole + 1) != "folder":
            return False
        session_id = bytes(data.data(MIME_SESSION_ID)).decode()
        folder_id = target.data(Qt.ItemDataRole.UserRole)
        self.session_moved.emit(session_id, folder_id)
        return True


class SettingsDialog(QDialog):
    def __init__(self, parent=None, enable_ollama=True):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(350, 150)
        
        layout = QVBoxLayout(self)
        
        self.ollama_checkbox = QCheckBox("Enable Smart Topic Detection (Ollama)")
        self.ollama_checkbox.setChecked(enable_ollama)
        self.ollama_checkbox.setToolTip("Uses local AI for highly accurate topic tracking.")
        layout.addWidget(self.ollama_checkbox)
        layout.addStretch()
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)


class EngineSignals(QObject):
    utterance_received = pyqtSignal(object)  # UtteranceRecord
    status_changed = pyqtSignal(str)
    drift_detected = pyqtSignal(float)       # similarity score
    similarity_updated = pyqtSignal(float)   # similarity score
    topic_inferred = pyqtSignal(str, str)    # (topic, source: 'ollama' | 'heuristic')
    ollama_status = pyqtSignal(object)       # ensure_ollama result dict

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intent")
        self.resize(1100, 700)

        self.dark = _is_dark_mode(QApplication.instance().palette())
        self._speaker_palette = SPEAKER_COLORS_DARK if self.dark else SPEAKER_COLORS_LIGHT
        self._muted_color = QColor("#64748b") if self.dark else QColor("#94a3b8")
        self._text_color = QColor("#f1f5f9") if self.dark else QColor("#1e293b")
        self._status_muted = "color: #64748b; font-weight: bold;" if self.dark else "color: #94a3b8; font-weight: bold;"

        self.engine_lock = threading.Lock()
        self.current_session_id: str | None = None
        self.signals = EngineSignals()
        self.signals.utterance_received.connect(self._on_utterance_gui)
        self.signals.status_changed.connect(self._on_status_gui)
        self.signals.drift_detected.connect(self._on_drift_gui)
        self.signals.similarity_updated.connect(self._on_similarity_gui)
        self.signals.topic_inferred.connect(self._on_topic_inferred_gui)
        self.signals.ollama_status.connect(self._on_ollama_status_gui)

        self.speaker_colors: dict[str, str] = {}
        self.speaker_idx = 0
        self._current_topic: str = ""

        self.duration_timer = QTimer(self)
        self.duration_timer.timeout.connect(self._update_duration)
        self.start_time = None
        self.elapsed_seconds = 0

        self.settings = self._load_settings()

        self._setup_ui()
        self._load_tree()

        # Preload engine so 'Start Session' is instant
        self.engine = TranscriberEngine(
            self._make_config(),
            on_utterance=self.signals.utterance_received.emit,
            on_status=self.signals.status_changed.emit,
            on_drift=self.signals.drift_detected.emit,
            on_similarity=self.signals.similarity_updated.emit,
        )
        # Clear Ollama model up-front if disabled
        if not self.settings.get("enable_ollama", True):
            self.engine.config = type(self.engine.config)(**{**vars(self.engine.config), "ollama_model": ""})

        threading.Thread(target=self._preload_model, daemon=True).start()

    def _load_settings(self) -> dict:
        try:
            with open("settings.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"enable_ollama": True}

    def _save_settings(self):
        with open("settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)

    def _preload_model(self):
        try:
            if self.settings.get("enable_ollama", True):
                # Check/start Ollama in the background and auto-detect model
                ollama_result = ensure_ollama(preferred_model=self.engine.config.ollama_model)
                # Update engine config with the detected model
                if ollama_result["model"]:
                    self.engine.config = type(self.engine.config)(
                        **{**vars(self.engine.config), "ollama_model": ollama_result["model"]}
                    )
                    print(f"Ollama model: {ollama_result['model']}", file=sys.stderr)
                self.signals.ollama_status.emit(ollama_result)
            else:
                self.signals.ollama_status.emit({"status": "disabled", "model": None})
                
            self.engine.load_model()
            
            # Warm up Ollama model so first topic inference is fast
            if self.settings.get("enable_ollama", True) and getattr(self, "engine", None) and getattr(self.engine, "config", None) and self.engine.config.ollama_model:
                from main import ask_ollama
                print("Warming up Ollama model...", file=sys.stderr)
                ask_ollama("Say OK", model=ollama_result["model"])
                print("Ollama Ready.", file=sys.stderr)
        except Exception as e:
            print(f"Error preloading: {e}")

    def _setup_ui(self):
        # Main layout: Splitter with Sidebar | Main View
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        sidebar_header = QLabel("Sessions")
        sidebar_header.setFont(QFont("Helvetica Neue", 14, QFont.Weight.Bold))
        sidebar_layout.addWidget(sidebar_header)

        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.tree_view.doubleClicked.connect(self._on_tree_double_click)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setDragDropMode(QTreeView.DragDropMode.DragDrop)
        self.tree_view.setDefaultDropAction(Qt.DropAction.MoveAction)

        self.tree_model = SessionTreeModel()
        self.tree_model.session_moved.connect(self._on_session_moved)
        self.tree_view.setModel(self.tree_model)
        sidebar_layout.addWidget(self.tree_view)

        new_folder_btn = QPushButton("New Folder")
        new_folder_btn.clicked.connect(self._action_new_folder)
        sidebar_layout.addWidget(new_folder_btn)
        
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self._open_settings)
        sidebar_layout.addWidget(settings_btn)

        # 2. Main View (Transcript)
        self.main_view = QWidget()
        main_view_layout = QVBoxLayout(self.main_view)
        main_view_layout.setContentsMargins(20, 20, 20, 20)

        # Top Bar
        top_bar = QHBoxLayout()
        self.title_label = QLabel("New Session")
        self.title_label.setFont(QFont("Helvetica Neue", 18, QFont.Weight.Bold))
        
        self.status_label = QLabel("Stopped")
        self.status_label.setStyleSheet(self._status_muted)

        self.duration_label = QLabel("00:00")
        self.duration_label.setStyleSheet(f"color: {self._muted_color.name()}; font-size: 14px;")

        self.start_btn = QPushButton("Start Session")
        self.start_btn.clicked.connect(self._start_session)
        self.start_btn.setMinimumHeight(35)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_session)
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setEnabled(False)

        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        top_bar.addWidget(self.duration_label)
        top_bar.addWidget(self.status_label)
        top_bar.addWidget(self.start_btn)
        top_bar.addWidget(self.stop_btn)
        
        main_view_layout.addLayout(top_bar)

        topic_bar = QHBoxLayout()
        self.topic_label = QLabel("")
        self.topic_label.setFont(QFont("Helvetica Neue", 13))
        self.topic_label.setStyleSheet(f"color: {self._muted_color.name()};")
        self.topic_label.setVisible(False)

        self.track_label = QLabel("")
        self.track_label.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        self.track_label.setVisible(False)

        self.source_label = QLabel("")
        self.source_label.setFont(QFont("Helvetica Neue", 11))
        self.source_label.setVisible(False)

        self.similarity_label = QLabel("")
        self.similarity_label.setFont(QFont("Helvetica Neue", 12))
        self.similarity_label.setStyleSheet(f"color: {self._muted_color.name()};")
        self.similarity_label.setVisible(False)

        topic_bar.addWidget(self.topic_label)
        topic_bar.addWidget(self.source_label)
        topic_bar.addStretch()
        topic_bar.addWidget(self.track_label)
        topic_bar.addWidget(self.similarity_label)
        main_view_layout.addLayout(topic_bar)

        drift_bg = "#fef2f2" if not self.dark else "#451a1a"
        drift_fg = "#dc2626" if not self.dark else "#fca5a5"
        drift_border = "#fecaca" if not self.dark else "#7f1d1d"
        self.drift_banner = QLabel("Off topic -- the conversation has drifted from the agenda")
        self.drift_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drift_banner.setStyleSheet(
            f"background-color: {drift_bg}; color: {drift_fg}; border: 1px solid {drift_border}; "
            "border-radius: 6px; padding: 8px; font-weight: bold; font-size: 13px;"
        )
        self.drift_banner.setVisible(False)
        main_view_layout.addWidget(self.drift_banner)

        # Ollama info banner (shown when Ollama is degraded)
        info_bg = "#eff6ff" if not self.dark else "#1e293b"
        info_fg = "#3b82f6" if not self.dark else "#93c5fd"
        info_border = "#bfdbfe" if not self.dark else "#1e3a5f"
        self.ollama_banner = QLabel("")
        self.ollama_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ollama_banner.setStyleSheet(
            f"background-color: {info_bg}; color: {info_fg}; border: 1px solid {info_border}; "
            "border-radius: 6px; padding: 8px; font-size: 12px;"
        )
        self.ollama_banner.setWordWrap(True)
        self.ollama_banner.setVisible(False)
        main_view_layout.addWidget(self.ollama_banner)

        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Helvetica Neue", 13))
        # Remove border for cleaner look
        self.transcript_area.setFrameShape(QFrame.Shape.NoFrame)
        main_view_layout.addWidget(self.transcript_area)

        # Splitter setup
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.main_view)
        splitter.setSizes([300, 800])

    # ---- Tree View / Database Logic ----
    
    def _load_tree(self):
        self.tree_model.clear()
        root = self.tree_model.invisibleRootItem()
        
        folders = db.list_folders()
        sessions = db.list_sessions()
        
        folder_items = {}
        
        # Add folders (root level for now, we can nest later if needed)
        drop_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDropEnabled
        drag_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled

        for f in folders:
            item = QStandardItem(f"📁 {f['name']}")
            item.setData(f['id'], Qt.ItemDataRole.UserRole)
            item.setData("folder", Qt.ItemDataRole.UserRole + 1)
            item.setFlags(drop_flags)
            folder_items[f['id']] = item
            root.appendRow(item)

        uncategorized = QStandardItem("📁 Uncategorized")
        uncategorized.setData(None, Qt.ItemDataRole.UserRole)
        uncategorized.setData("folder", Qt.ItemDataRole.UserRole + 1)
        uncategorized.setFlags(drop_flags)
        root.appendRow(uncategorized)

        for s in sessions:
            title = s['title'] or "Untitled Session"
            date_str = datetime.fromisoformat(s['started_at']).strftime("%b %d, %H:%M")
            item = QStandardItem(f"🎙 {title} ({date_str})")
            item.setData(s['id'], Qt.ItemDataRole.UserRole)
            item.setData("session", Qt.ItemDataRole.UserRole + 1)
            item.setFlags(drag_flags)

            f_id = s.get('folder_id')
            if f_id and f_id in folder_items:
                folder_items[f_id].appendRow(item)
            else:
                uncategorized.appendRow(item)
                
        self.tree_view.expandAll()

    def _on_tree_context_menu(self, position):
        index = self.tree_view.indexAt(position)
        menu = QMenu()
        
        if index.isValid():
            item_type = self.tree_model.itemFromIndex(index).data(Qt.ItemDataRole.UserRole + 1)
            item_id = self.tree_model.itemFromIndex(index).data(Qt.ItemDataRole.UserRole)
            
            if item_type == "folder" and item_id: # Not the fake 'Uncategorized' folder
                rename_action = menu.addAction("Rename Folder")
                rename_action.triggered.connect(lambda: self._action_rename_folder(item_id, index))
                
                delete_action = menu.addAction("Delete Folder")
                delete_action.triggered.connect(lambda: self._action_delete_folder(item_id))
                
            elif item_type == "session":
                rename_action = menu.addAction("Rename Session")
                rename_action.triggered.connect(lambda: self._action_rename_session(item_id, index))
                
                # Move to folder submenu
                move_menu = menu.addMenu("Move to Folder")
                for f in db.list_folders():
                    action = move_menu.addAction(f['name'])
                    action.triggered.connect(lambda checked, fid=f['id']: self._action_move_session(item_id, fid))
                root_action = move_menu.addAction("Uncategorized")
                root_action.triggered.connect(lambda: self._action_move_session(item_id, None))
                
                delete_action = menu.addAction("Delete Session")
                delete_action.triggered.connect(lambda: self._action_delete_session(item_id))

        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def _on_tree_double_click(self, index: QModelIndex):
        item = self.tree_model.itemFromIndex(index)
        if item.data(Qt.ItemDataRole.UserRole + 1) == "session":
            session_id = item.data(Qt.ItemDataRole.UserRole)
            self._load_session(session_id)

    def _on_session_moved(self, session_id: str, folder_id):
        db.move_session_to_folder(session_id, folder_id)
        self._load_tree()

    # ---- Actions ----

    def _action_new_folder(self):
        text, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and text:
            db.create_folder(text)
            self._load_tree()
            
    def _action_rename_folder(self, folder_id, index):
        old_name = self.tree_model.itemFromIndex(index).text().replace("📁 ", "")
        text, ok = QInputDialog.getText(self, "Rename Folder", "New folder name:", text=old_name)
        if ok and text:
            db.rename_folder(folder_id, text)
            self._load_tree()

    def _action_delete_folder(self, folder_id):
        reply = QMessageBox.question(self, "Confirm Delete", "Delete this folder? Sessions inside will be uncategorized.", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            db.delete_folder(folder_id)
            self._load_tree()

    def _action_rename_session(self, session_id, index):
        old_title = db.get_session(session_id)['title']
        text, ok = QInputDialog.getText(self, "Rename Session", "New title:", text=old_title)
        if ok and text:
            db.update_session_title(session_id, text)
            self._load_tree()
            if self.current_session_id == session_id:
                self.title_label.setText(text)

    def _action_delete_session(self, session_id):
        reply = QMessageBox.question(self, "Confirm Delete", "Delete this session permanently?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            db.delete_session(session_id)
            if self.current_session_id == session_id:
                self.transcript_area.clear()
                self.title_label.setText("New Session")
                self.current_session_id = None
            self._load_tree()

    def _action_move_session(self, session_id, folder_id):
        db.move_session_to_folder(session_id, folder_id)
        self._load_tree()

    def _load_settings(self) -> dict:
        try:
            with open("settings.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {} # Default settings
        except json.JSONDecodeError:
            print("Warning: Could not decode settings.json. Using default settings.")
            return {}

    def _save_settings(self):
        with open("settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)

    def _open_settings(self):
        dialog = SettingsDialog(self, enable_ollama=self.settings.get("enable_ollama", True))
        if dialog.exec():
            # If user clicked OK
            new_enable = dialog.ollama_checkbox.isChecked()
            if new_enable != self.settings.get("enable_ollama", True):
                self.settings["enable_ollama"] = new_enable
                self._save_settings()
                QMessageBox.information(
                    self,
                    "Restart Required",
                    "Settings saved. Please restart the application for changes to take effect."
                )

    # ---- Engine Integration ----

    def _make_config(self) -> AppConfig:
        from main import (
            DEFAULT_MODEL, DEFAULT_BACKEND_DEVICE, DEFAULT_COMPUTE_TYPE,
            DEFAULT_CPU_THREADS, DEFAULT_BEAM_SIZE, DEFAULT_BEST_OF, DEFAULT_PATIENCE,
            DEFAULT_SAMPLE_RATE, DEFAULT_BLOCK_MS, DEFAULT_CALIBRATION_SECONDS,
            DEFAULT_SILENCE_SECONDS, DEFAULT_PRE_ROLL_SECONDS, DEFAULT_MAX_UTTERANCE_SECONDS,
            DEFAULT_FORCE_SPLIT_OVERLAP_SECONDS, DEFAULT_MIN_UTTERANCE_SECONDS,
            DEFAULT_ENERGY_FLOOR, DEFAULT_SPEECH_RATIO, DEFAULT_EXACT_CONTEXT_CHARS,
            DEFAULT_RECENT_CONTEXT_CHARS, DEFAULT_MEMORY_BUDGET_CHARS,
            DEFAULT_PROMPT_TAIL_CHARS, DEFAULT_PROMPT_BUDGET_CHARS, DEFAULT_HALLUCINATION_FILTER,
            DEFAULT_SPEAKER_THRESHOLD
        )
        output_path, metadata_path = default_output_paths(None, None)
        return AppConfig(
            model=DEFAULT_MODEL,
            backend_device=DEFAULT_BACKEND_DEVICE,
            compute_type=DEFAULT_COMPUTE_TYPE,
            cpu_threads=DEFAULT_CPU_THREADS,
            beam_size=DEFAULT_BEAM_SIZE,
            best_of=DEFAULT_BEST_OF,
            patience=DEFAULT_PATIENCE,
            output_path=output_path,
            metadata_path=metadata_path,
            language="en",
            notes="",
            device=None,
            sample_rate=DEFAULT_SAMPLE_RATE,
            block_ms=DEFAULT_BLOCK_MS,
            calibration_seconds=DEFAULT_CALIBRATION_SECONDS,
            silence_seconds=DEFAULT_SILENCE_SECONDS,
            pre_roll_seconds=DEFAULT_PRE_ROLL_SECONDS,
            max_utterance_seconds=DEFAULT_MAX_UTTERANCE_SECONDS,
            force_split_overlap_seconds=DEFAULT_FORCE_SPLIT_OVERLAP_SECONDS,
            min_utterance_seconds=DEFAULT_MIN_UTTERANCE_SECONDS,
            energy_floor=DEFAULT_ENERGY_FLOOR,
            speech_ratio=DEFAULT_SPEECH_RATIO,
            exact_context_chars=DEFAULT_EXACT_CONTEXT_CHARS,
            recent_context_chars=DEFAULT_RECENT_CONTEXT_CHARS,
            memory_budget_chars=DEFAULT_MEMORY_BUDGET_CHARS,
            prompt_tail_chars=DEFAULT_PROMPT_TAIL_CHARS,
            prompt_budget_chars=DEFAULT_PROMPT_BUDGET_CHARS,
            hallucination_filter=DEFAULT_HALLUCINATION_FILTER,
            diarize=False,
            speaker_threshold=DEFAULT_SPEAKER_THRESHOLD,
            speakers_file=None,
            ollama_model="llama3.2:1b",
            warmup_only=False,
            verbose_model=False,
        )

    def _start_session(self):
        with self.engine_lock:
            if self.engine and self.engine.is_running:
                return

            self.transcript_area.clear()
            self.speaker_colors.clear()
            self.speaker_idx = 0
            self.drift_banner.setVisible(False)
            self._current_topic = ""

            self.topic_label.setText("Topic: detecting...")
            self.topic_label.setStyleSheet(f"color: {self._muted_color.name()};")
            self.topic_label.setVisible(True)
            self.source_label.setText("")
            self.source_label.setVisible(False)
            self.track_label.setText("")
            self.track_label.setVisible(True)
            self.similarity_label.setText("")
            self.similarity_label.setVisible(True)

            selected = self.tree_view.selectedIndexes()
            folder_id = None
            if selected:
                item = self.tree_model.itemFromIndex(selected[0])
                if item.data(Qt.ItemDataRole.UserRole + 1) == "folder":
                    folder_id = item.data(Qt.ItemDataRole.UserRole)
                elif item.data(Qt.ItemDataRole.UserRole + 1) == "session":
                    sess = db.get_session(item.data(Qt.ItemDataRole.UserRole))
                    folder_id = sess.get('folder_id')

            self.current_session_id = db.create_session(
                model=self.engine.config.model,
                folder_id=folder_id
            )
            self.title_label.setText("Recording...")

            def _start():
                try:
                    import time
                    while self.engine.model is None:
                        time.sleep(0.1)
                    self.engine.enable_topic_tracking(
                        on_topic_inferred=self.signals.topic_inferred.emit,
                    )
                    self.engine.start()
                except Exception as exc:
                    self.signals.status_changed.emit(f"error: {exc}")

            self.start_btn.setEnabled(False)
            self.start_btn.setText("Starting...")
            threading.Thread(target=_start, daemon=True).start()

    def _stop_session(self):
        with self.engine_lock:
            if not self.engine:
                return
                
            def _stop():
                try:
                    self.engine.stop()
                    if self.current_session_id:
                        db.end_session(
                            self.current_session_id,
                            self.engine.get_transcript(),
                            self.engine.get_compressed_memory()
                        )
                        # Auto-title
                        utts = self.engine.get_utterances()
                        if utts:
                            db.update_session_title(self.current_session_id, utts[0].text[:60])
                except Exception as exc:
                    print(f"Stop error: {exc}")
                finally:
                    # Refresh tree on main thread
                    QTimer.singleShot(0, self._load_tree)

            self.stop_btn.setEnabled(False)
            threading.Thread(target=_stop, daemon=True).start()

    def _load_session(self, session_id):
        # Prevent loading while recording
        if self.engine and self.engine.is_running:
            QMessageBox.warning(self, "Recording", "Please stop the current recording first.")
            return
            
        session = db.get_session(session_id)
        if not session:
            return
            
        self.current_session_id = session_id
        self.title_label.setText(session['title'] or "Untitled Session")
        self.transcript_area.clear()
        self.speaker_colors.clear()
        self.speaker_idx = 0
        
        self.elapsed_seconds = session.get('duration_seconds', 0)
        self.duration_label.setText(self._format_time(self.elapsed_seconds))
        
        self.status_label.setText("Archived")
        self.status_label.setStyleSheet(self._status_muted)
        
        # Render utterances
        for u in session['utterances']:
            # Create dummy UtteranceRecord for rendering
            record = UtteranceRecord(
                index=u['index'], start_seconds=u['start_seconds'], end_seconds=u['end_seconds'],
                text=u['text'], raw_text="", speaker=u['speaker'], forced_split=False
            )
            self._on_utterance_gui(record, save_to_db=False)

    # ---- Callbacks (Running on Main Thread) ----

    def _on_status_gui(self, status: str):
        if status.startswith("error"):
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Session")
            return
            
        if status == "listening":
            self.status_label.setText("Listening")
            self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.start_btn.setText("Start Session")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Start timer
            self.start_time = datetime.now()
            self.elapsed_seconds = 0
            self.duration_timer.start(1000)
            
            # Refresh tree to show new session
            self._load_tree()
            
        elif status == "stopped":
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet(self._status_muted)
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Session")
            self.stop_btn.setEnabled(False)
            self.duration_timer.stop()
            self.drift_banner.setVisible(False)
            self.track_label.setVisible(False)
            self.similarity_label.setVisible(False)

    def _on_utterance_gui(self, record: UtteranceRecord, save_to_db=True):
        if save_to_db and self.current_session_id:
            db.save_utterance(
                self.current_session_id, record.index, record.start_seconds, record.end_seconds,
                record.text, record.raw_text, record.speaker, record.forced_split
            )
            
        # Render to text area
        cursor = self.transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        time_str = f"[{self._format_time(record.start_seconds)}] "
        
        fmt = QTextCharFormat()
        fmt.setForeground(QBrush(self._muted_color))
        cursor.insertText(time_str, fmt)

        if record.speaker:
            if record.speaker not in self.speaker_colors:
                self.speaker_colors[record.speaker] = self._speaker_palette[self.speaker_idx % len(self._speaker_palette)]
                self.speaker_idx += 1

            speaker_fmt = QTextCharFormat()
            speaker_fmt.setForeground(QBrush(QColor(self.speaker_colors[record.speaker])))
            speaker_fmt.setFontWeight(QFont.Weight.Bold)
            cursor.insertText(f"[{record.speaker}] ", speaker_fmt)

        text_fmt = QTextCharFormat()
        text_fmt.setForeground(QBrush(self._text_color))
        cursor.insertText(f"{record.text}\n", text_fmt)
        
        self.transcript_area.setTextCursor(cursor)
        self.transcript_area.ensureCursorVisible()

    def _on_topic_inferred_gui(self, topic: str, source: str = "heuristic"):
        self._current_topic = topic
        self.topic_label.setText(f"Topic: {topic}")
        self.title_label.setText(topic[:60])
        if source == "ollama":
            ollama_color = "#10b981" if self.dark else "#059669"
            self.source_label.setText("⚡ Ollama")
            self.source_label.setStyleSheet(f"color: {ollama_color}; font-size: 11px;")
        else:
            heuristic_color = "#f59e0b" if self.dark else "#d97706"
            self.source_label.setText("◆ Heuristic")
            self.source_label.setStyleSheet(f"color: {heuristic_color}; font-size: 11px;")
        self.source_label.setVisible(True)

    def _on_ollama_status_gui(self, result: dict):
        status = result.get("status", "unavailable")
        model = result.get("model")
        if status in ("ready", "started") and model:
            self.ollama_banner.setVisible(False)
            if status == "started":
                print(f"Ollama auto-started with model: {model}", flush=True)
        elif status == "not_installed":
            self.ollama_banner.setText(
                "ℹ️  Ollama is not installed — topic classification will use basic heuristics. "
                "Install from ollama.com for smarter topic detection."
            )
            self.ollama_banner.setVisible(True)
        elif status in ("ready", "started") and not model:
            self.ollama_banner.setText(
                "ℹ️  Ollama is running but no models are installed. "
                "Run 'ollama pull deepseek-r1' to enable smart topic detection."
            )
            self.ollama_banner.setVisible(True)
        elif status == "disabled":
            self.ollama_banner.setText(
                "ℹ️  Ollama topic detection is disabled in Settings. Using basic heuristics."
            )
            self.ollama_banner.setVisible(True)
        else:  # unavailable
            self.ollama_banner.setText(
                "ℹ️  Could not reach Ollama — topic classification will use basic heuristics. "
                "Run 'ollama serve' for smarter topic detection."
            )
            self.ollama_banner.setVisible(True)

    def _on_drift_gui(self, similarity: float):
        self.drift_banner.setVisible(True)
        self.track_label.setText("OFF TRACK")
        self.track_label.setStyleSheet("color: #ef4444; font-weight: bold;")

    def _on_similarity_gui(self, similarity: float):
        pct = int(similarity * 100)
        if similarity >= 0.5:
            color = "#10b981" if self.dark else "#059669"
            self.track_label.setText("On Track")
            self.track_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        elif similarity >= 0.35:
            color = "#f59e0b" if self.dark else "#d97706"
            self.track_label.setText("Drifting...")
            self.track_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            color = "#ef4444"
        self.similarity_label.setText(f"{pct}%")
        self.similarity_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if similarity >= 0.35:
            self.drift_banner.setVisible(False)

    def _update_duration(self):
        if self.start_time:
            self.elapsed_seconds = int((datetime.now() - self.start_time).total_seconds())
            self.duration_label.setText(self._format_time(self.elapsed_seconds))

    def _format_time(self, seconds: float) -> str:
        s = int(seconds)
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"


def main():
    app = QApplication(sys.argv)
    dark = _is_dark_mode(app.palette())
    app.setStyleSheet(_build_stylesheet(dark))

    db.init_db()

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

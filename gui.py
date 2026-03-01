import sys
import threading
import json
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeView, QTextEdit, QPushButton, QLabel, QSplitter, QMenu,
    QInputDialog, QMessageBox, QFrame, QHeaderView, QStyleFactory, QDialog,
    QCheckBox, QDialogButtonBox, QLineEdit, QTabWidget,
)
from PyQt6.QtGui import (
    QAction, QIcon, QFont, QColor, QPalette, QTextCursor, QTextDocument,
    QStandardItemModel, QStandardItem, QTextCharFormat, QBrush
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QModelIndex, QMimeData

import database as db
from main import TranscriberEngine, AppConfig, default_output_paths, UtteranceRecord, ensure_ollama
from concept_map_widget import ConceptMapWidget
from concept_extraction import extract_concepts, merge_concept_maps

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
                background-color: #1a36b4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #11247a; }
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
            background-color: #1a36b4;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #11247a; }
        QPushButton:disabled { background-color: #e2e8f0; color: #94a3b8; }
        QLabel { color: #1e293b; }
        QSplitter::handle { background-color: #e2e8f0; width: 1px; }
        QMenu { background-color: #ffffff; color: #1e293b; border: 1px solid #e2e8f0; }
        QMenu::item:selected { background-color: #e0e7ff; }
    """

MIME_SESSION_ID = "application/x-intent-session-id"
MIME_FOLDER_ID = "application/x-intent-folder-id"


class SessionTreeModel(QStandardItemModel):
    """Custom model that allows dragging sessions and folders onto folders."""

    session_moved = pyqtSignal(str, object)  # session_id, folder_id (str or None)
    folder_moved = pyqtSignal(str, object)   # folder_id, parent_id (str or None)

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        return [MIME_SESSION_ID, MIME_FOLDER_ID]

    def mimeData(self, indexes):
        mime = QMimeData()
        for index in indexes:
            item = self.itemFromIndex(index)
            if not item: continue
            if item.data(Qt.ItemDataRole.UserRole + 1) == "session":
                mime.setData(MIME_SESSION_ID, item.data(Qt.ItemDataRole.UserRole).encode())
                return mime
            elif item.data(Qt.ItemDataRole.UserRole + 1) == "folder":
                f_id = item.data(Qt.ItemDataRole.UserRole)
                if f_id: # Don't drag Uncategorized
                    mime.setData(MIME_FOLDER_ID, f_id.encode())
                    return mime
        return mime

    def canDropMimeData(self, data, action, row, column, parent):
        if not (data.hasFormat(MIME_SESSION_ID) or data.hasFormat(MIME_FOLDER_ID)):
            return False
        
        target = self.itemFromIndex(parent)
        # Drop onto background -> moving to root (None folder_id)
        if target is None:
            return True
        
        if target.data(Qt.ItemDataRole.UserRole + 1) != "folder":
            return False
            
        # Prevention: Folder cannot be dropped into itself or its descendants
        if data.hasFormat(MIME_FOLDER_ID):
            moving_id = bytes(data.data(MIME_FOLDER_ID)).decode()
            curr = target
            while curr:
                if curr.data(Qt.ItemDataRole.UserRole) == moving_id:
                    return False # Cycle detected
                curr = curr.parent()
        
        return True

    def dropMimeData(self, data, action, row, column, parent):
        target = self.itemFromIndex(parent)
        folder_id = target.data(Qt.ItemDataRole.UserRole) if target else None
        
        if data.hasFormat(MIME_SESSION_ID):
            session_id = bytes(data.data(MIME_SESSION_ID)).decode()
            self.session_moved.emit(session_id, folder_id)
            return True
        elif data.hasFormat(MIME_FOLDER_ID):
            moving_folder_id = bytes(data.data(MIME_FOLDER_ID)).decode()
            self.folder_moved.emit(moving_folder_id, folder_id)
            return True
            
        return False


class SettingsDialog(QDialog):
    def __init__(self, parent=None, enable_ollama=True, deepgram_enabled=False, deepgram_api_key="", llm_provider="ollama", gemini_api_key="", gemini_model="gemini-1.5-flash"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(420, 340)
        
        layout = QVBoxLayout(self)
        
        # --- AI Topic Detection Section ---
        ollama_header = QLabel("AI Topic Detection")
        ollama_header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(ollama_header)
        
        self.ollama_checkbox = QCheckBox("Enable AI Topic Detection")
        self.ollama_checkbox.setChecked(enable_ollama)
        self.ollama_checkbox.setToolTip("Use AI to infer topics and detect drift.")
        layout.addWidget(self.ollama_checkbox)
        
        # --- LLM Provider Section ---
        llm_header = QLabel("AI Provider")
        llm_header.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        layout.addWidget(llm_header)
        
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        self.provider_group = QButtonGroup(self)
        
        provider_layout = QHBoxLayout()
        self.radio_ollama = QRadioButton("Local AI (Ollama)")
        self.radio_gemini = QRadioButton("Gemini AI (Cloud)")
        
        self.provider_group.addButton(self.radio_ollama)
        self.provider_group.addButton(self.radio_gemini)
        
        provider_layout.addWidget(self.radio_ollama)
        provider_layout.addWidget(self.radio_gemini)
        layout.addLayout(provider_layout)
        
        gemini_key_layout = QHBoxLayout()
        gemini_key_label = QLabel("Gemini API Key:")
        gemini_key_layout.addWidget(gemini_key_label)
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_input.setPlaceholderText("Paste your Gemini API key...")
        self.gemini_key_input.setText(gemini_api_key)
        gemini_key_layout.addWidget(self.gemini_key_input)
        layout.addLayout(gemini_key_layout)
        
        gemini_model_layout = QHBoxLayout()
        gemini_model_label = QLabel("Gemini Model:")
        gemini_model_layout.addWidget(gemini_model_label)
        self.gemini_model_input = QLineEdit()
        self.gemini_model_input.setPlaceholderText("e.g. gemini-1.5-flash")
        self.gemini_model_input.setText(gemini_model)
        gemini_model_layout.addWidget(self.gemini_model_input)
        layout.addLayout(gemini_model_layout)
        
        # Set initial values
        if llm_provider == "gemini":
            self.radio_gemini.setChecked(True)
            self.gemini_key_input.setEnabled(True)
        else:
            self.radio_ollama.setChecked(True)
            self.gemini_key_input.setEnabled(False)
            
        self.radio_gemini.toggled.connect(self.gemini_key_input.setEnabled)
        self.radio_gemini.toggled.connect(self.gemini_model_input.setEnabled)
        
        # --- Separator ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # --- Deepgram Section ---
        dg_header = QLabel("Cloud Transcription")
        dg_header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(dg_header)
        
        self.deepgram_checkbox = QCheckBox("Use Deepgram Cloud Transcription")
        self.deepgram_checkbox.setChecked(deepgram_enabled)
        self.deepgram_checkbox.setToolTip("Uses Deepgram's Nova-3 model for superior accuracy with accents, overlap, and diarization.")
        layout.addWidget(self.deepgram_checkbox)
        
        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        key_layout.addWidget(key_label)
        self.deepgram_key_input = QLineEdit()
        self.deepgram_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepgram_key_input.setPlaceholderText("Paste your Deepgram API key...")
        self.deepgram_key_input.setText(deepgram_api_key)
        self.deepgram_key_input.setEnabled(deepgram_enabled)
        key_layout.addWidget(self.deepgram_key_input)
        layout.addLayout(key_layout)
        
        # Toggle API key field visibility based on checkbox
        self.deepgram_checkbox.toggled.connect(self.deepgram_key_input.setEnabled)
        
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
        self.engine = None  # set before UI/load_tree to avoid early attribute access
        self.signals = EngineSignals()
        self.signals.utterance_received.connect(self._on_utterance_gui)
        self.signals.status_changed.connect(self._on_status_gui)
        self.signals.drift_detected.connect(self._on_drift_gui)
        self.signals.similarity_updated.connect(self._on_similarity_gui)
        self.signals.topic_inferred.connect(self._on_topic_inferred_gui)
        self.signals.ollama_status.connect(self._on_ollama_status_gui)

        # Signal for concept map results (thread-safe)
        self._map_ready_signal = pyqtSignal  # placeholder
        self.signals.status_changed.connect(self._check_map_result)  # reuse status for simplicity
        self._pending_map_graph = None
        self._target_index_to_select = None

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
            return {"enable_ollama": True, "last_folder_id": None}

    def _save_settings(self):
        with open("settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)

    def _preload_model(self):
        try:
            provider = self.settings.get("llm_provider", "ollama")
            use_ollama = self.settings.get("enable_ollama", True) and provider == "ollama"

            if provider == "gemini":
                gem_key = self.settings.get("gemini_api_key", "").strip()
                if not gem_key:
                    self.signals.ollama_status.emit({"status": "no_key", "model": None})
                else:
                    self.signals.ollama_status.emit({"status": "gemini", "model": "gemini"})
            elif use_ollama:
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
            if use_ollama and getattr(self, "engine", None) and getattr(self.engine, "config", None) and self.engine.config.ollama_model:
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
        self.tree_view.clicked.connect(self._on_tree_clicked)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setDragDropMode(QTreeView.DragDropMode.DragDrop)
        self.tree_view.setDefaultDropAction(Qt.DropAction.MoveAction)
        
        # Enable complete row selection and keep highlight active
        self.tree_view.setSelectionMode(QTreeView.SelectionMode.SingleSelection)
        self.tree_view.setSelectionBehavior(QTreeView.SelectionBehavior.SelectRows)
        self.tree_view.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Remove dotted outline

        self.tree_model = SessionTreeModel()
        self.tree_model.session_moved.connect(self._on_session_moved)
        self.tree_model.folder_moved.connect(self._on_folder_moved)
        self.tree_view.setModel(self.tree_model)
        sidebar_layout.addWidget(self.tree_view)

        new_folder_btn = QPushButton("+ New Folder")
        new_folder_btn.clicked.connect(self._action_new_folder)
        sidebar_layout.addWidget(new_folder_btn)
        
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        btn_bg = "#334155" if self.dark else "#e2e8f0"
        btn_fg = "#e2e8f0" if self.dark else "#475569"
        btn_hover = "#475569" if self.dark else "#cbd5e1"
        settings_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {btn_bg}; color: {btn_fg}; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {btn_hover}; }}
        """)
        sidebar_layout.addWidget(settings_btn)

        # 2. Main View (Transcript)
        self.main_view = QWidget()
        main_view_layout = QVBoxLayout(self.main_view)
        main_view_layout.setContentsMargins(20, 20, 20, 20)

        # Top Bar
        top_bar = QHBoxLayout()
        self.title_input = QLineEdit("New Session")
        self.title_input.setPlaceholderText("Set Manual Topic...")
        self.title_input.setFont(QFont("Helvetica Neue", 18, QFont.Weight.Bold))
        # Style QLineEdit to look like a label but editable
        bg_color = "transparent"
        text_color = self._text_color.name()
        hover_bg = "#f1f5f9" if not self.dark else "#334155"
        self.title_input.setStyleSheet(f"""
            QLineEdit {{
                background: {bg_color};
                color: {text_color};
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 2px 5px;
            }}
            QLineEdit:hover {{
                background: {hover_bg};
            }}
            QLineEdit:focus {{
                border: 1px solid #3b82f6;
                background: {bg_color};
            }}
        """)
        
        self.status_label = QLabel("Stopped")
        self.status_label.setStyleSheet(self._status_muted)

        self.duration_label = QLabel("00:00")
        self.duration_label.setStyleSheet(f"color: {self._muted_color.name()}; font-size: 14px;")

        self.start_btn = QPushButton("Start Session")
        self.start_btn.clicked.connect(self._start_session)
        self.start_btn.setFixedHeight(35)

        self.new_session_btn = QPushButton("+ New Session")
        self.new_session_btn.clicked.connect(self._prepare_new_session)
        self.new_session_btn.setFixedHeight(35)
        new_bg = "#1a36b4"
        new_hover = "#2563eb"
        self.new_session_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {new_bg}; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {new_hover}; }}
        """)
        self.new_session_btn.setVisible(False)

        self.map_btn = QPushButton("Generate Map")
        self.map_btn.clicked.connect(self._generate_concept_map)
        self.map_btn.setFixedHeight(35)
        map_bg = "#8b5cf6"
        map_hover = "#7c3aed"
        self.map_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {map_bg}; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {map_hover}; }}
            QPushButton:disabled {{ background-color: {'#334155' if self.dark else '#e2e8f0'}; color: {'#64748b' if self.dark else '#94a3b8'}; }}
        """)
        self.map_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_session)
        self.stop_btn.setFixedHeight(35)
        self.stop_btn.setEnabled(False)
        stop_bg = "#dc2626"
        stop_hover = "#b91c1c"
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {stop_bg}; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {stop_hover}; }}
            QPushButton:disabled {{ background-color: {"#334155" if self.dark else "#e2e8f0"}; color: {"#64748b" if self.dark else "#94a3b8"}; }}
        """)

        top_bar.addWidget(self.title_input, stretch=1)
        top_bar.addStretch()
        top_bar.addWidget(self.duration_label)
        top_bar.addWidget(self.status_label)
        top_bar.addWidget(self.new_session_btn)
        top_bar.addWidget(self.map_btn)
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
        self._was_off_track = False  # Track recovery state

        self.source_label = QLabel("")
        self.source_label.setFont(QFont("Helvetica Neue", 11))
        self.source_label.setVisible(False)

        topic_bar.addWidget(self.topic_label)
        topic_bar.addWidget(self.source_label)
        topic_bar.addStretch()
        topic_bar.addWidget(self.track_label)
        main_view_layout.addLayout(topic_bar)

        drift_bg = "#fef2f2" if not self.dark else "#451a1a"
        drift_fg = "#dc2626" if not self.dark else "#fca5a5"
        drift_border = "#fecaca" if not self.dark else "#7f1d1d"
        self.drift_banner = QLabel("The conversation appears to have drifted from the original topic.")
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

        # Search bar for transcript
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcript...")
        self.search_input.setStyleSheet(
            f"background: {'#334155' if self.dark else '#f1f5f9'}; "
            f"color: {'#e2e8f0' if self.dark else '#1e293b'}; "
            f"border: 1px solid {'#475569' if self.dark else '#cbd5e1'}; "
            "border-radius: 4px; padding: 4px;"
        )
        self.search_input.returnPressed.connect(lambda: self._search_transcript(forward=True))
        
        self.search_prev_btn = QPushButton("Prev")
        self.search_prev_btn.clicked.connect(lambda: self._search_transcript(forward=False))
        
        self.search_next_btn = QPushButton("Next")
        self.search_next_btn.clicked.connect(lambda: self._search_transcript(forward=True))
        
        self.search_count_label = QLabel("")
        self.search_count_label.setStyleSheet(f"color: {'#94a3b8' if self.dark else '#64748b'}; font-size: 12px;")
        
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_count_label)
        search_layout.addWidget(self.search_prev_btn)
        search_layout.addWidget(self.search_next_btn)

        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Helvetica Neue", 13))
        self.transcript_area.setFrameShape(QFrame.Shape.NoFrame)

        # Concept map widget
        self.concept_map = ConceptMapWidget(dark=self.dark)

        # Tabbed view: Transcript | Concept Map
        self.tab_widget = QTabWidget()
        tab_bg = '#1e293b' if self.dark else '#ffffff'
        tab_fg = '#e2e8f0' if self.dark else '#1e293b'
        tab_sel = '#1a36b4'
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{
                background: {'#334155' if self.dark else '#e2e8f0'};
                color: {tab_fg};
                padding: 8px 0px;
                min-width: 150px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                text-align: center;
            }}
            QTabBar::tab:selected {{ background: {tab_sel}; color: white; }}
            QTabBar::tab:hover {{ background: {'#475569' if self.dark else '#cbd5e1'}; }}
        """)
        transcript_widget = QWidget()
        transcript_layout = QVBoxLayout(transcript_widget)
        transcript_layout.addLayout(search_layout)
        transcript_layout.addWidget(self.transcript_area)

        # Chat Search UI
        chat_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask LLM about this transcript or folder...")
        self.chat_input.setStyleSheet(self.search_input.styleSheet())
        self.chat_input.returnPressed.connect(self._ask_llm_about_transcript)
        
        self.chat_send_btn = QPushButton("Send")
        self.chat_send_btn.clicked.connect(self._ask_llm_about_transcript)
        self.chat_send_btn.setStyleSheet(f"""
            QPushButton {{ background-color: #1a36b4; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-weight: bold; }}
            QPushButton:hover {{ background-color: #11247a; }}
            QPushButton:disabled {{ background-color: {'#334155' if self.dark else '#e2e8f0'}; color: {'#64748b' if self.dark else '#94a3b8'}; }}
        """)
        
        chat_layout.addWidget(self.chat_input)
        chat_layout.addWidget(self.chat_send_btn)
        
        self.chat_output_area = QTextEdit()
        self.chat_output_area.setReadOnly(True)
        self.chat_output_area.setFont(QFont("Helvetica Neue", 12))
        self.chat_output_area.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_output_area.setStyleSheet(f"background: {'#1e293b' if self.dark else '#f8fafc'}; color: {'#94a3b8' if self.dark else '#475569'};")
        self.chat_output_area.setMaximumHeight(150)
        self.chat_output_area.setVisible(False)
        
        transcript_layout.addWidget(self.chat_output_area)
        transcript_layout.addLayout(chat_layout)

        self.tab_widget.addTab(transcript_widget, "Transcript")
        self.tab_widget.addTab(self.concept_map, "Concept Map")
        main_view_layout.addWidget(self.tab_widget)

        # Splitter setup
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.main_view)
        splitter.setSizes([300, 800])

    # ---- Tree View / Database Logic ----
    
    def _load_tree(self):
        try:
            self.tree_model.clear()
            root = self.tree_model.invisibleRootItem()
            
            folders = db.list_folders()
            sessions = db.list_sessions()
            
            folder_items = {}
            last_folder_id = self.settings.get("last_folder_id")
            self._target_index_to_select = None
            
            # Add folders (root level for now, we can nest later if needed)
            drop_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDropEnabled
            drag_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled

            folder_font = QFont("Helvetica Neue", 14, QFont.Weight.Bold)
            session_font = QFont("Helvetica Neue", 13)

            # 1. Create all folder items first
            for f in folders:
                item = QStandardItem(f["name"])
                item.setFont(folder_font)
                item.setData(f['id'], Qt.ItemDataRole.UserRole)
                item.setData("folder", Qt.ItemDataRole.UserRole + 1)
                item.setFlags(drop_flags | drag_flags) 
                folder_items[f['id']] = item

            # 2. Build folder hierarchy
            for f in folders:
                item = folder_items[f['id']]
                p_id = f.get('parent_id')
                if p_id and p_id in folder_items:
                    folder_items[p_id].appendRow(item)
                else:
                    root.appendRow(item)

            # 3. Add Uncategorized (always root level)
            uncategorized = QStandardItem("Uncategorized")
            uncategorized.setFont(folder_font)
            uncategorized.setData(None, Qt.ItemDataRole.UserRole)
            uncategorized.setData("folder", Qt.ItemDataRole.UserRole + 1)
            uncategorized.setFlags(drop_flags)
            root.appendRow(uncategorized)
            
            if last_folder_id is None:
                self._target_index_to_select = uncategorized.index()

            for s in sessions:
                title = s['title'] or "Untitled Session"
                date_str = datetime.fromisoformat(s['started_at']).strftime("%b %d, %H:%M")
                item = QStandardItem(f"{title}  ·  {date_str}")
                item.setFont(session_font)
                item.setData(s['id'], Qt.ItemDataRole.UserRole)
                item.setData("session", Qt.ItemDataRole.UserRole + 1)
                item.setFlags(drag_flags)

                f_id = s.get('folder_id')
                if f_id and f_id in folder_items:
                    folder_items[f_id].appendRow(item)
                else:
                    uncategorized.appendRow(item)

            # 5. Handle empty folders and selection
            def _post_process(parent_item):
                item_type = parent_item.data(Qt.ItemDataRole.UserRole + 1)
                if item_type == "folder":
                    f_id = parent_item.data(Qt.ItemDataRole.UserRole)
                    if str(f_id) == str(last_folder_id):
                        self._target_index_to_select = parent_item.index()

                    if parent_item.rowCount() == 0:
                        placeholder = QStandardItem("(empty)")
                        placeholder.setForeground(QBrush(self._muted_color))
                        placeholder.setFont(QFont("Helvetica Neue", 11))
                        placeholder.setData("placeholder", Qt.ItemDataRole.UserRole + 1)
                        parent_item.appendRow(placeholder)
                    else:
                        for i in range(parent_item.rowCount()):
                            _post_process(parent_item.child(i))

            for i in range(root.rowCount()):
                _post_process(root.child(i))
                    
            self.tree_view.expandAll()
            
            if getattr(self, "current_session_id", None) is None and not (getattr(self, "engine", None) and self.engine.is_running):
                if self._target_index_to_select and self._target_index_to_select.isValid():
                    self.tree_view.setCurrentIndex(self._target_index_to_select)
                    self._on_tree_clicked(self._target_index_to_select)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading session tree: {e}", file=sys.stderr)

    def _on_tree_context_menu(self, position):
        index = self.tree_view.indexAt(position)
        menu = QMenu()
        
        if index.isValid():
            item_type = self.tree_model.itemFromIndex(index).data(Qt.ItemDataRole.UserRole + 1)
            item_id = self.tree_model.itemFromIndex(index).data(Qt.ItemDataRole.UserRole)
            
            if item_type == "folder":
                generate_action = menu.addAction("Generate Concept Map")
                generate_action.triggered.connect(lambda: self._generate_folder_map(item_id, self.tree_model.itemFromIndex(index).text()))
                menu.addSeparator()
                
                if item_id is not None:
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
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        item_id = item.data(Qt.ItemDataRole.UserRole)
        if item_type == "session":
            self._load_session(item_id)
        elif item_type == "folder":
            self._generate_folder_map(item_id, item.text())
            
    def _on_tree_clicked(self, index: QModelIndex):
        item = self.tree_model.itemFromIndex(index)
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        item_id = item.data(Qt.ItemDataRole.UserRole)
        
        # Prevent loading while recording
        if self.engine and self.engine.is_running:
            return
            
        if item_type == "folder":
            sessions = db.get_folder_session_transcripts(item_id)
            if sessions:
                merged = "\n\n".join([f"--- Session: {s['title']} ---\n{s['transcript']}" for s in sessions if s.get('transcript')]).strip()
            else:
                merged = "No transcripts found in this folder."
                
            if merged:
                self.current_session_id = None
                self.title_input.setText(f"Folder: {item.text()}")
                self.transcript_area.setPlainText(merged)
                self.status_label.setText("Folder View")
                self.status_label.setStyleSheet(self._status_muted)
                
                self.start_btn.setVisible(False)
                self.stop_btn.setVisible(False)
                self.new_session_btn.setVisible(True)
                self.map_btn.setEnabled(True)
                self.concept_map.clear_map()
                self.duration_label.setText("")
                
                # Save as last viewed folder
                self.settings["last_folder_id"] = item_id
                self._save_settings()

    def _on_session_moved(self, session_id: str, folder_id):
        try:
            db.move_session_to_folder(session_id, folder_id)
            self._load_tree()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move session:\n{str(e)}")

    def _on_folder_moved(self, folder_id: str, parent_id: str | None):
        try:
            db.update_folder_parent(folder_id, parent_id)
            self._load_tree()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move folder:\n{str(e)}")

    # ---- Actions ----

    def _action_new_folder(self):
        text, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and text:
            try:
                db.create_folder(text)
                self._load_tree()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create folder:\n{str(e)}")
            
    def _action_rename_folder(self, folder_id, index):
        old_name = self.tree_model.itemFromIndex(index).text()
        text, ok = QInputDialog.getText(self, "Rename Folder", "New folder name:", text=old_name)
        if ok and text:
            try:
                db.rename_folder(folder_id, text)
                self._load_tree()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to rename folder:\n{str(e)}")

    def _action_delete_folder(self, folder_id):
        reply = QMessageBox.question(self, "Confirm Delete", "Delete this folder? Sessions inside will be uncategorized.", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                db.delete_folder(folder_id)
                self._load_tree()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete folder:\n{str(e)}")

    def _action_rename_session(self, session_id, index):
        try:
            old_title = db.get_session(session_id).get('title', "Session")
        except Exception:
            old_title = "Session"
            
        text, ok = QInputDialog.getText(self, "Rename Session", "New title:", text=old_title)
        if ok and text:
            try:
                db.update_session_title(session_id, text)
                self._load_tree()
                if self.current_session_id == session_id:
                    self.title_input.setText(text)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to rename session:\n{str(e)}")

    def _action_delete_session(self, session_id):
        reply = QMessageBox.question(self, "Confirm Delete", "Delete this session permanently?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                db.delete_session(session_id)
                # Delete corresponding audio folder
                folder_path = Path("transcripts") / session_id
                if folder_path.exists():
                    import shutil
                    shutil.rmtree(folder_path, ignore_errors=True)
                    
                if self.current_session_id == session_id:
                    self.transcript_area.clear()
                    self.title_input.setText("New Session")
                    self.title_input.setEnabled(True)
                    self.current_session_id = None
                self._load_tree()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete session:\n{str(e)}")

    def _action_move_session(self, session_id, folder_id):
        try:
            db.move_session_to_folder(session_id, folder_id)
            self._load_tree()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move session:\n{str(e)}")


    def _open_settings(self):
        dialog = SettingsDialog(
            self,
            enable_ollama=self.settings.get("enable_ollama", True),
            deepgram_enabled=self.settings.get("deepgram_enabled", False),
            deepgram_api_key=self.settings.get("deepgram_api_key", ""),
            llm_provider=self.settings.get("llm_provider", "ollama"),
            gemini_api_key=self.settings.get("gemini_api_key", ""),
            gemini_model=self.settings.get("gemini_model", "gemini-1.5-flash"),
        )
        if dialog.exec():
            changed = False
            # Ollama
            new_ollama = dialog.ollama_checkbox.isChecked()
            if new_ollama != self.settings.get("enable_ollama", True):
                self.settings["enable_ollama"] = new_ollama
                changed = True
            # LLM Provider
            new_provider = "gemini" if dialog.radio_gemini.isChecked() else "ollama"
            if new_provider != self.settings.get("llm_provider", "ollama"):
                self.settings["llm_provider"] = new_provider
                changed = True
            # Gemini Key
            new_gem_key = dialog.gemini_key_input.text().strip()
            if new_gem_key != self.settings.get("gemini_api_key", ""):
                self.settings["gemini_api_key"] = new_gem_key
                changed = True
            # Deepgram
            new_dg = dialog.deepgram_checkbox.isChecked()
            new_key = dialog.deepgram_key_input.text().strip()
            if new_dg != self.settings.get("deepgram_enabled", False):
                self.settings["deepgram_enabled"] = new_dg
                changed = True
            if new_key != self.settings.get("deepgram_api_key", ""):
                self.settings["deepgram_api_key"] = new_key
                changed = True
            # Gemini Model
            new_gem_model = dialog.gemini_model_input.text().strip()
            if new_gem_model != self.settings.get("gemini_model", "gemini-1.5-flash"):
                self.settings["gemini_model"] = new_gem_model
                changed = True
            
            if changed:
                self._save_settings()
                # Update engine config immediately if it exists
                if self.engine:
                    self.engine.config = self._make_config()
                
                self.status_label.setText("Settings Updated")
                QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))

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
            deepgram_enabled=self.settings.get("deepgram_enabled", False),
            deepgram_api_key=self.settings.get("deepgram_api_key", ""),
            llm_provider=self.settings.get("llm_provider", "ollama"),
            gemini_api_key=self.settings.get("gemini_api_key", ""),
            gemini_model=self.settings.get("gemini_model", "gemini-1.5-flash"),
        )

    def _start_session(self):
        with self.engine_lock:
            if self.engine and self.engine.is_running:
                return

            # Refresh config from settings before starting
            self.engine.config = self._make_config()

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

            initial_topic = self.title_input.text().strip()
            if initial_topic == "New Session" or not initial_topic:
                initial_topic = None
                
            if not self.current_session_id:
                selected = self.tree_view.selectedIndexes()
                folder_id = None
                if selected:
                    item = self.tree_model.itemFromIndex(selected[0])
                    if item.data(Qt.ItemDataRole.UserRole + 1) == "folder":
                        folder_id = item.data(Qt.ItemDataRole.UserRole)
    
                self.current_session_id = db.create_session(
                    model=self.engine.config.model,
                    folder_id=folder_id
                )
            self.title_input.setText(initial_topic or "Recording...")

            def _start():
                try:
                    import time
                    while self.engine.model is None:
                        time.sleep(0.1)
                    self.engine.enable_topic_tracking(
                        on_topic_inferred=self.signals.topic_inferred.emit,
                        initial_topic=initial_topic or "New Session"
                    )
                    self.engine.start()
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self.signals.status_changed.emit(f"error: {exc}")

            self.start_btn.setEnabled(False)
            self.start_btn.setText("Starting...")
            self.title_input.setEnabled(False) # Disable title input when starting
            self.new_session_btn.setVisible(False)
            self.map_btn.setEnabled(False)
            threading.Thread(target=_start, daemon=True).start()

    def _prepare_new_session(self):
        """Reset the UI for a new recording session."""
        if self.engine and self.engine.is_running:
            QMessageBox.warning(self, "Recording", "Please stop the current recording first.")
            return

        self.current_session_id = None
        self.title_input.setText("New Session")
        self.title_input.setEnabled(True)
        self.transcript_area.clear()
        self.speaker_colors.clear()
        self.speaker_idx = 0
        self.elapsed_seconds = 0
        self.duration_label.setText("00:00")
        
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet(self._status_muted)
        
        self.concept_map.clear_map()
        self.tab_widget.setCurrentIndex(0)
        
        # Reset buttons
        self.new_session_btn.setVisible(False)
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setVisible(True)
        self.stop_btn.setEnabled(False)
        self.map_btn.setEnabled(False)
        
        # Clear tree selection
        self.tree_view.clearSelection()

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
                        # Auto-title ONLY if it doesn't already have one
                        sess = db.get_session(self.current_session_id)
                        if sess and (not sess['title'] or sess['title'] in ["New Session", "Recording...", "Untitled Session", "Started"]):
                            final_transcript = self.engine.get_transcript()
                            if final_transcript.strip():
                                # Ask LLM for a title in the background
                                model_config = getattr(self.engine, 'config', None)
                                ollama_model = model_config.ollama_model if model_config else "deepseek-r1:8b"
                                
                                def _generate_title(sid, txt):
                                    prompt = f"Analyze this transcript and write a concise 3-5 word title for the meeting. Output ONLY the title text, nothing else.\n\nTRANSCRIPT:\n{txt}"
                                    try:
                                        llm_provider = getattr(model_config, 'llm_provider', 'ollama')
                                        if llm_provider == "gemini":
                                            from main import ask_gemini
                                            gemini_key = getattr(model_config, 'gemini_api_key', '')
                                            title = ask_gemini(prompt, api_key=gemini_key)
                                        else:
                                            from main import ask_ollama
                                            title = ask_ollama(prompt, model=ollama_model)
                                            
                                        if title:
                                            title = title.strip(' "\'')
                                            # Strip out `<think>` tags if the model leaked them
                                            import re
                                            title = re.sub(r'<think>.*?</think>', '', title, flags=re.DOTALL).strip()
                                            
                                            db.update_session_title(sid, title)
                                            # Refresh to show new title
                                            self.signals.status_changed.emit("__refresh_tree__")
                                    except Exception as e:
                                        print(f"Auto-title failed: {e}", file=sys.stderr)
                                        
                                threading.Thread(target=_generate_title, args=(self.current_session_id, final_transcript), daemon=True).start()
                            else:
                                db.update_session_title(self.current_session_id, "Empty Session")

                        self.current_session_id = None
                        
                    self.signals.status_changed.emit("stopped")
                except Exception as exc:
                    print(f"Stop error: {exc}")
                finally:
                    # Refresh tree on main thread
                    QTimer.singleShot(0, self._load_tree)

            self.stop_btn.setEnabled(False)
            self.title_input.setEnabled(True) # Re-enable title input when stopping
            threading.Thread(target=_stop, daemon=True).start()

    def _load_session(self, session_id):
        # Prevent loading while recording
        if self.engine and self.engine.is_running:
            QMessageBox.warning(self, "Recording", "Please stop the current recording first.")
            return
        
        try:
            session = db.get_session(session_id)
            if not session:
                return
                
            self.current_session_id = session_id
            self.title_input.setText(session['title'] or "Untitled Session")
            self.transcript_area.clear()
            self.speaker_colors.clear()
            self.speaker_idx = 0
            
            self.elapsed_seconds = session.get('duration_seconds', 0)
            self.duration_label.setText(self._format_time(self.elapsed_seconds))
            
            self.status_label.setText("Archived")
            self.status_label.setStyleSheet(self._status_muted)
            
            # Switch buttons for archived map
            self.start_btn.setVisible(False)
            self.stop_btn.setVisible(False)
            self.new_session_btn.setVisible(True)
            self.map_btn.setEnabled(True)
            
            # Render utterances
            for u in session['utterances']:
                record = UtteranceRecord(
                    index=u['index'], start_seconds=u['start_seconds'], end_seconds=u['end_seconds'],
                    text=u['text'], raw_text="", speaker=u['speaker'], forced_split=False
                )
                self._on_utterance_gui(record, save_to_db=False)
            
            # Load existing concept map if available
            existing_map = db.get_concept_map(session_id=session_id)
            if existing_map:
                self.concept_map.load_graph(existing_map['graph'])
            else:
                self.concept_map.clear_map()
                
        except Exception as e:
            print(f"Error loading session: {e}", file=sys.stderr)
            QMessageBox.warning(self, "Error", f"Failed to load session:\n{str(e)}")

    def _generate_concept_map(self):
        """Generate a concept map for the current session."""
        if not self.current_session_id:
            return
        
        session = db.get_session(self.current_session_id)
        if not session:
            return
        
        # Use saved transcript, or build from utterances as fallback
        transcript = session.get('transcript', '').strip()
        if not transcript and session.get('utterances'):
            transcript = "\n".join(
                f"[{u.get('speaker', 'SPEAKER')}] {u['text']}"
                for u in session['utterances'] if u.get('text', '').strip()
            )
        
        if not transcript:
            QMessageBox.information(self, "No Content", "This session has no transcript content to map.")
            return
        
        self.map_btn.setEnabled(False)
        self.map_btn.setText("Generating...")
        self.concept_map.show_generating("Sending transcript to Ollama...")
        self.tab_widget.setCurrentIndex(1)  # Switch to Concept Map tab
        session_id = self.current_session_id
        model_config = getattr(self.engine, 'config', None)
        ollama_model = model_config.ollama_model if model_config else "deepseek-r1:8b"
        llm_provider = model_config.llm_provider if model_config else "ollama"
        gemini_key = model_config.gemini_api_key if model_config else ""
        
        def _do_generate():
            try:
                self.signals.status_changed.emit("__map_status__Analyzing transcript with LLM...")
                
                # Callback to pipe stream thoughts to GUI
                def _on_status(msg: str):
                    self.signals.status_changed.emit(f"__map_status__{msg}")
                    
                gemini_model = self.settings.get("gemini_model", "gemini-1.5-flash")
                
                graph = extract_concepts(
                    transcript, 
                    model=ollama_model,
                    status_callback=_on_status,
                    llm_provider=llm_provider,
                    api_key=gemini_key,
                    gemini_model=gemini_model
                )
                
                self.signals.status_changed.emit("__map_status__Saving concept map...")
                # Save to database
                db.save_concept_map(
                    graph_json=json.dumps(graph),
                    session_id=session_id,
                )
                self._pending_map_graph = graph
                # Signal the GUI thread
                self.signals.status_changed.emit("__map_ready__")
            except Exception as e:
                print(f"Concept map generation failed: {e}", file=sys.stderr)
                self.signals.status_changed.emit("__map_error__")
        
        threading.Thread(target=_do_generate, daemon=True).start()

    def _check_map_result(self, status: str):
        """Handle concept map generation result on the GUI thread."""
        if status.startswith("__map_status__"):
            msg = status[len("__map_status__"):]
            self.concept_map.update_status(msg)
            return
        if status == "__map_ready__" or status == "__folder_map_ready__":
            graph = self._pending_map_graph
            self._pending_map_graph = None
            if graph:
                self.concept_map.load_graph(graph)
                self.tab_widget.setCurrentIndex(1)  # Switch to Concept Map tab
            self.map_btn.setEnabled(bool(self.current_session_id))
            self.map_btn.setText("Generate Map")
            if status == "__folder_map_ready__":
                # Update title to remove generating status
                current = self.title_input.text()
                self.title_input.setText(current.replace(" (Generating...)", " (Folder Map)"))
        elif status == "__map_error__":
            self.map_btn.setEnabled(bool(self.current_session_id))
            self.map_btn.setText("Generate Map")
            self.concept_map.clear_map()
            QMessageBox.warning(self, "Error", "Failed to generate concept map. Is Ollama running?")

    def _search_transcript(self, forward: bool = True):
        """Search the transcript text and highlight matches."""
        query = self.search_input.text()
        if not query:
            self.search_count_label.setText("")
            self.transcript_area.setExtraSelections([])
            return

        # Count total occurrences
        text = self.transcript_area.toPlainText()
        total_matches = text.lower().count(query.lower())
        
        if total_matches == 0:
            self.search_count_label.setText("0 of 0")
            # Change background to red briefly to indicate not found
            self.search_input.setStyleSheet(
                f"background: {'#7f1d1d' if self.dark else '#fee2e2'}; "
                f"color: {'#fef2f2' if self.dark else '#991b1b'}; "
                f"border: 1px solid {'#ef4444' if self.dark else '#f87171'}; "
                "border-radius: 4px; padding: 4px;"
            )
            QTimer.singleShot(500, lambda: self.search_input.setStyleSheet(
                f"background: {'#334155' if self.dark else '#f1f5f9'}; "
                f"color: {'#e2e8f0' if self.dark else '#1e293b'}; "
                f"border: 1px solid {'#475569' if self.dark else '#cbd5e1'}; "
                "border-radius: 4px; padding: 4px;"
            ))
            self.transcript_area.setExtraSelections([])
            return

        # Find flag configurations
        flags = QTextDocument.FindFlag(0)  # No flags by default
        if not forward:
            flags |= QTextDocument.FindFlag.FindBackward

        found = self.transcript_area.find(query, flags)
        
        # Wrap around if not found
        if not found:
            cursor = self.transcript_area.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.End if not forward else QTextCursor.MoveOperation.Start
            )
            self.transcript_area.setTextCursor(cursor)
            found = self.transcript_area.find(query, flags)

        if found:
            # Calculate current position index
            cursor = self.transcript_area.textCursor()
            pos = cursor.selectionStart()
            # Count occurrences *before* this match
            current_idx = text[:pos].lower().count(query.lower()) + 1
            self.search_count_label.setText(f"{current_idx} of {total_matches}")

            # Ensure the active match is visible
            self.transcript_area.ensureCursorVisible()
            
            # Find and highlight ALL matches in the document
            extra_selections = []
            
            # 1. Active match (Blue)
            active_selection = QTextEdit.ExtraSelection()
            active_bg = QColor("#0369a1" if self.dark else "#bae6fd")
            active_fg = QColor("#ffffff" if self.dark else "#0f172a")
            active_selection.format.setBackground(active_bg)
            active_selection.format.setForeground(active_fg)
            active_selection.cursor = cursor
            extra_selections.append(active_selection)
            
            # 2. All other matches (Yellow)
            bg_color = QColor("#ca8a04" if self.dark else "#fef08a")
            fg_color = QColor("#ffffff" if self.dark else "#854d0e")
            
            doc = self.transcript_area.document()
            highlight_cursor = QTextCursor(doc)
            highlight_cursor.movePosition(QTextCursor.MoveOperation.Start)
            
            while True:
                highlight_cursor = doc.find(query, highlight_cursor, flags)
                if highlight_cursor.isNull():
                    break
                
                # Skip the active selection since we already colored it differently
                if highlight_cursor.selectionStart() != cursor.selectionStart():
                    selection = QTextEdit.ExtraSelection()
                    selection.format.setBackground(bg_color)
                    selection.format.setForeground(fg_color)
                    selection.cursor = highlight_cursor
                    extra_selections.append(selection)
            
            self.transcript_area.setExtraSelections(extra_selections)

    def _generate_folder_map(self, folder_id: str, folder_name: str):
        """Generate a merged concept map for all sessions in a folder."""
        # Switch buttons for archived map
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(False)
        self.new_session_btn.setVisible(True)
        
        # Check for existing
        existing = db.get_concept_map(folder_id=folder_id)
        if existing:
            self.concept_map.load_graph(existing['graph'])
            self.title_input.setText(f"{folder_name} (Folder Map)")
            self.tab_widget.setCurrentIndex(1)
            self.map_btn.setEnabled(False)
            return
        
        sessions = db.get_folder_session_transcripts(folder_id)
        if not sessions:
            QMessageBox.information(self, "Empty Folder", "This folder has no sessions with transcripts.")
            return

        self.title_input.setText(f"{folder_name} (Generating...)")
        model_config = getattr(self.engine, 'config', None)
        ollama_model = model_config.ollama_model if model_config else "deepseek-r1:8b"
        llm_provider = model_config.llm_provider if model_config else "ollama"
        gemini_key = model_config.gemini_api_key if model_config else ""
        
        def _do_folder_map():
            try:
                def _on_status(msg: str):
                    self.signals.status_changed.emit(f"__map_status__{msg}")
                
                # Generate maps for each session that doesn't have one
                session_maps = []
                for i, s in enumerate(sessions):
                    existing_map = db.get_concept_map(session_id=s['id'])
                    if existing_map:
                        session_maps.append({
                            "session_id": s['id'],
                            "title": s['title'],
                            "graph": existing_map['graph'],
                        })
                    else:
                        gemini_model = self.settings.get("gemini_model", "gemini-1.5-flash")
                        graph = extract_concepts(
                            s['transcript'], 
                            model=ollama_model, 
                            status_callback=_on_status, 
                            llm_provider=llm_provider, 
                            api_key=gemini_key,
                            gemini_model=gemini_model
                        )
                        db.save_concept_map(graph_json=json.dumps(graph), session_id=s['id'])
                        session_maps.append({
                            "session_id": s['id'],
                            "title": s['title'],
                            "graph": graph,
                        })
                
                # Merge
                self.signals.status_changed.emit("__map_status__Merging all folder sessions...")
                merged = merge_concept_maps(
                    session_maps, 
                    model=ollama_model, 
                    status_callback=_on_status, 
                    llm_provider=llm_provider, 
                    api_key=gemini_key,
                    gemini_model=gemini_model
                )
                db.save_concept_map(graph_json=json.dumps(merged), folder_id=folder_id)
                self._pending_map_graph = merged
                self.signals.status_changed.emit("__folder_map_ready__")
            except Exception as e:
                print(f"Folder map generation failed: {e}", file=sys.stderr)
                self.signals.status_changed.emit("__map_error__")
        
        threading.Thread(target=_do_folder_map, daemon=True).start()

    def _ask_llm_about_transcript(self):
        """Send the current transcript and user question to the LLM."""
        question = self.chat_input.text().strip()
        transcript = self.transcript_area.toPlainText().strip()
        
        # If the text area is empty, check if a folder is selected to use as context
        if not transcript:
            indexes = self.tree_view.selectedIndexes()
            if indexes:
                index = indexes[0]
                item_type = self.tree_model.itemFromIndex(index).data(Qt.ItemDataRole.UserRole + 1)
                item_id = self.tree_model.itemFromIndex(index).data(Qt.ItemDataRole.UserRole)
                if item_type == "folder":
                    sessions = db.get_folder_session_transcripts(item_id)
                    if sessions:
                        transcript = "\n\n".join([f"--- Session: {s['title']} ---\n{s['transcript']}" for s in sessions if s.get('transcript')]).strip()
        
        if not question or not transcript:
            if not transcript:
                QMessageBox.information(self, "No Context", "Please select a session or folder with a transcript to ask questions about.")
            return
            
        self.chat_send_btn.setEnabled(False)
        self.chat_output_area.setVisible(True)
        self.chat_output_area.clear()
        
        # Format user question to show up instantly
        user_color = "#3b82f6" if self.dark else "#2563eb"
        
        model = getattr(self.engine, 'config', None)
        llm_provider = model.llm_provider if model and hasattr(model, 'llm_provider') else "ollama"
        ai_name = "Gemini" if llm_provider == "gemini" else "Ollama"
        
        self.chat_output_area.append(f"<span style='color: {user_color}; font-weight: bold;'>You:</span> {question}")
        self.chat_output_area.append(f"<span style='color: #8b5cf6; font-weight: bold;'>{ai_name}:</span> ")
        
        self.chat_input.clear()
        
        prompt = f"""You are a helpful assistant answering a question about a transcript.
Use only the information provided in the transcript below to answer. If the answer is not in the transcript, say so.

TRANSCRIPT:
{transcript}

QUESTION:
{question}
"""
        def _stream_answer():
            try:
                if llm_provider == "gemini":
                    from main import ask_gemini_stream
                    gemini_key = model.gemini_api_key if model else ""
                    gemini_model = model.gemini_model if model else "gemini-1.5-flash"
                    for chunk in ask_gemini_stream(prompt, api_key=gemini_key, model=gemini_model):
                        self.signals.status_changed.emit(f"__chat_chunk__{chunk}")
                else:
                    from main import ask_ollama_stream
                    ollama_model = model.ollama_model if model else "deepseek-r1:8b"
                    for chunk in ask_ollama_stream(prompt, model=ollama_model):
                        self.signals.status_changed.emit(f"__chat_chunk__{chunk}")
            except Exception as e:
                self.signals.status_changed.emit(f"__chat_error__{str(e)}")
            finally:
                self.signals.status_changed.emit("__chat_done__")
                
        threading.Thread(target=_stream_answer, daemon=True).start()

    # ---- Callbacks (Running on Main Thread) ----

    def _on_status_gui(self, status: str):
        # Handle chat signal chunks
        if status.startswith("__chat_chunk__"):
            chunk = status[len("__chat_chunk__"):]
            # Ensure we append to the same line by inserting plain text at the end
            cursor = self.chat_output_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(chunk)
            self.chat_output_area.ensureCursorVisible()
            return
        elif status.startswith("__chat_error__"):
            err = status[len("__chat_error__"):]
            self.chat_output_area.append(f"<br><span style='color: #ef4444;'>Error: {err}</span>")
            return
        elif status == "__chat_done__":
            self.chat_send_btn.setEnabled(True)
            self.chat_output_area.append("<br><br>") # padding for next question
            return
            
        # Handle concept map signals
        if status.startswith("__map_status__") or status.startswith("__folder_map_") or status.startswith("__map_error"):
            self._check_map_result(status)
            return
            
        if status == "__refresh_tree__":
            self._load_tree()
            return
            
        if status.startswith("error"):
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Session")
            self.title_input.setEnabled(True) # Re-enable title input on error
            return
            
        if status == "listening":
            self.status_label.setText("Listening")
            self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.start_btn.setText("Start Session")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.title_input.setEnabled(False) # Disable title input when listening
            
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

    def _on_utterance_gui(self, record: UtteranceRecord, save_to_db=True):
        try:
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
        except Exception as e:
            print(f"Error rendering utterance: {e}", file=sys.stderr)

    def _on_topic_inferred_gui(self, topic: str, source: str = "heuristic"):
        try:
            self._current_topic = topic
            self.topic_label.setText(f"Topic: {topic}")
            # Only overwrite the title_input if the user hasn't manually set one this session
            if self.title_input.text() in ("New Session", "Recording...", ""):
                self.title_input.setText(topic[:60])
                if self.current_session_id:
                    db.update_session_title(self.current_session_id, topic[:60])
                    self._load_tree()
            if source == "ollama":
                ollama_color = "#10b981" if self.dark else "#059669"
                self.source_label.setText("LLM")
                self.source_label.setStyleSheet(f"color: {ollama_color}; font-size: 11px; font-weight: bold;")
            else:
                heuristic_color = "#f59e0b" if self.dark else "#d97706"
                self.source_label.setText("Heuristic")
                self.source_label.setStyleSheet(f"color: {heuristic_color}; font-size: 11px; font-weight: bold;")
            self.source_label.setVisible(True)
        except Exception as e:
            print(f"Error updating topic display: {e}", file=sys.stderr)

    def _on_ollama_status_gui(self, result: dict):
        status = result.get("status", "unavailable")
        model = result.get("model")
        if status == "gemini":
            self.ollama_banner.setText("Using Gemini for topic detection.")
            self.ollama_banner.setVisible(False)
        elif status == "no_key":
            self.ollama_banner.setText(
                "Gemini topic detection selected but no API key is set. Using basic heuristics."
            )
            self.ollama_banner.setVisible(True)
        elif status in ("ready", "started") and model:
            self.ollama_banner.setVisible(False)
            if status == "started":
                print(f"Ollama auto-started with model: {model}", flush=True)
        elif status == "not_installed":
            self.ollama_banner.setText(
                "Ollama is not installed. Topic classification will use basic heuristics. "
                "Install from ollama.com for smarter topic detection."
            )
            self.ollama_banner.setVisible(True)
        elif status in ("ready", "started") and not model:
            self.ollama_banner.setText(
                "Ollama is running but no models are installed. "
                "Run 'ollama pull deepseek-r1' to enable smart topic detection."
            )
            self.ollama_banner.setVisible(True)
        elif status == "disabled":
            self.ollama_banner.setText(
                "AI topic detection is disabled in Settings. Using basic heuristics."
            )
            self.ollama_banner.setVisible(True)
        else:  # unavailable
            self.ollama_banner.setText(
                "Could not reach Ollama. Topic classification will use basic heuristics. "
                "Run 'ollama serve' for smarter topic detection."
            )
            self.ollama_banner.setVisible(True)

    def _on_drift_gui(self, similarity: float):
        self.drift_banner.setVisible(True)
        self.track_label.setText("Off Track")
        self.track_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        self._was_off_track = True

    def _on_similarity_gui(self, similarity: float):
        if similarity >= 0.5:
            color = "#10b981" if self.dark else "#059669"
            if self._was_off_track:
                self.track_label.setText("Recovering")
                recover_color = "#3b82f6" if self.dark else "#2563eb"
                self.track_label.setStyleSheet(f"color: {recover_color}; font-weight: bold;")
                self._was_off_track = False
            else:
                self.track_label.setText("On Track")
                self.track_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.drift_banner.setVisible(False)
        elif similarity >= 0.35:
            color = "#f59e0b" if self.dark else "#d97706"
            self.track_label.setText("Drifting")
            self.track_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.drift_banner.setVisible(False)
        else:
            self.track_label.setText("Off Track")
            self.track_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self._was_off_track = True

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

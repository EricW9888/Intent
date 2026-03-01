"""Interactive concept map widget using QGraphicsView with force-directed layout."""
from __future__ import annotations

import math
import random
from typing import Any

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsRectItem, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGraphicsItem,
)
from PyQt6.QtGui import (
    QColor, QPen, QBrush, QFont, QPainterPath, QPolygonF,
    QPainter, QWheelEvent, QMouseEvent,
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer


# Node type → color mapping
NODE_COLORS = {
    "person":   {"bg": "#2563eb", "fg": "#ffffff"},  # Blue
    "topic":    {"bg": "#7c3aed", "fg": "#ffffff"},  # Purple
    "decision": {"bg": "#059669", "fg": "#ffffff"},  # Green
    "action":   {"bg": "#e11d48", "fg": "#ffffff"},  # Red
    "question": {"bg": "#d97706", "fg": "#ffffff"},  # Amber
    "concept":  {"bg": "#4f46e5", "fg": "#ffffff"},  # Indigo
}

NODE_COLORS_DARK = {
    "person":   {"bg": "#3b82f6", "fg": "#ffffff"},
    "topic":    {"bg": "#8b5cf6", "fg": "#ffffff"},
    "decision": {"bg": "#10b981", "fg": "#ffffff"},
    "action":   {"bg": "#ef4444", "fg": "#ffffff"},
    "question": {"bg": "#f59e0b", "fg": "#ffffff"},
    "concept":  {"bg": "#6366f1", "fg": "#ffffff"},
}


class ConceptNode(QGraphicsRectItem):
    """A single concept node rendered as a rounded rectangle with a label."""

    def __init__(self, node_data: dict, dark: bool = True):
        super().__init__()
        self.node_id = node_data["id"]
        self.node_label = node_data["label"]
        self.node_type = node_data.get("type", "concept")
        self.source_session = node_data.get("source_session", "")
        self.dark = dark

        palette = (NODE_COLORS_DARK if dark else NODE_COLORS).get(
            self.node_type, NODE_COLORS_DARK["concept"] if dark else NODE_COLORS["concept"]
        )
        self.base_color = QColor(palette["bg"])
        self.text_color = QColor(palette["fg"])

        # Size based on label length
        char_width = max(len(self.node_label) * 8, 80)
        self.node_width = min(char_width + 24, 200)
        self.node_height = 36

        self.setRect(0, 0, self.node_width, self.node_height)
        self.setBrush(QBrush(self.base_color))
        self.setPen(QPen(self.base_color.darker(120), 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(10)

        # Label
        self._label = QGraphicsTextItem(self.node_label, self)
        self._label.setDefaultTextColor(self.text_color)
        font = QFont("Helvetica Neue", 14, QFont.Weight.Bold)
        self._label.setFont(font)
        # Center label
        text_rect = self._label.boundingRect()
        self._label.setPos(
            (self.node_width - text_rect.width()) / 2,
            (self.node_height - text_rect.height()) / 2,
        )

        # Type badge
        type_label = self.node_type.upper()
        self._type_badge = QGraphicsTextItem(type_label, self)
        badge_font = QFont("Helvetica Neue", 10, QFont.Weight.Bold)
        self._type_badge.setFont(badge_font)
        self._type_badge.setDefaultTextColor(
            self.text_color if not dark else QColor("#94a3b8")
        )
        badge_rect = self._type_badge.boundingRect()
        self._type_badge.setPos(
            (self.node_width - badge_rect.width()) / 2,
            self.node_height + 2,
        )
        self._type_badge.setVisible(False)

        self.edges: list[ConceptEdge] = []
        self._highlighted = False
        self._dimmed = False

        # Tooltip
        tip = f"{self.node_label}\nType: {self.node_type}"
        if self.source_session:
            tip += f"\nSession: {self.source_session}"
        self.setToolTip(tip)

    def paint(self, painter, option, widget=None):
        """Draw a rounded rectangle instead of a sharp one."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        if self._dimmed:
            color = QColor(self.base_color)
            color.setAlpha(60 if self.dark else 100)
            painter.setBrush(QBrush(color))
            pen_color = QColor(self.base_color)
            pen_color.setAlpha(100 if self.dark else 140)
            painter.setPen(QPen(pen_color, 1))
        elif self._highlighted:
            painter.setBrush(QBrush(self.base_color.lighter(130)))
            painter.setPen(QPen(QColor("#ffffff") if self.dark else QColor("#000000"), 3))
        else:
            painter.setBrush(QBrush(self.base_color))
            painter.setPen(QPen(self.base_color.darker(110), 2))

        painter.drawRoundedRect(rect, 8, 8)

    def set_highlighted(self, highlighted: bool):
        self._highlighted = highlighted
        self._dimmed = False
        self._type_badge.setVisible(highlighted)
        self._label.setDefaultTextColor(self.text_color)
        self.update()

    def set_dimmed(self, dimmed: bool):
        self._dimmed = dimmed
        self._highlighted = False
        self._type_badge.setVisible(False)
        if dimmed:
            dim_color = self.text_color
            dim_color.setAlpha(60)
            self._label.setDefaultTextColor(dim_color)
        else:
            self._label.setDefaultTextColor(self.text_color)
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        if not self._dimmed:
            self.setBrush(QBrush(self.base_color.lighter(115)))
            self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self._dimmed and not self._highlighted:
            self.setBrush(QBrush(self.base_color))
            self.update()
        super().hoverLeaveEvent(event)


class ConceptEdge(QGraphicsPathItem):
    """An edge between two concept nodes, drawn as a curved line with an arrowhead."""

    def __init__(self, source: ConceptNode, target: ConceptNode, label: str = "", dark: bool = True):
        super().__init__()
        self.source = source
        self.target = target
        self.edge_label = label
        self.dark = dark

        color = QColor("#94a3b8") if dark else QColor("#64748b")
        self.base_color = color
        self.setPen(QPen(color, 1.5))
        self.setZValue(1)

        # Edge label
        self._label_item = None
        if label:
            self._label_item = QGraphicsTextItem(label, self)
            font = QFont("Helvetica Neue", 12, QFont.Weight.Bold)
            self._label_item.setFont(font)
            self._label_item.setDefaultTextColor(
                QColor("#cbd5e1") if dark else QColor("#334155")
            )

        self._dimmed = False
        self._highlighted = False

        source.edges.append(self)
        target.edges.append(self)
        self.update_position()

    def update_position(self):
        """Recalculate the path between source and target centers."""
        src = self.source.sceneBoundingRect().center()
        tgt = self.target.sceneBoundingRect().center()

        # Simple curved path
        path = QPainterPath(src)
        dx = tgt.x() - src.x()
        dy = tgt.y() - src.y()
        ctrl1 = QPointF(src.x() + dx * 0.3, src.y() + dy * 0.1)
        ctrl2 = QPointF(src.x() + dx * 0.7, src.y() + dy * 0.9)
        path.cubicTo(ctrl1, ctrl2, tgt)
        self.setPath(path)

        # Position label at midpoint
        if self._label_item:
            mid = QPointF(src.x() + dx * 0.5, src.y() + dy * 0.5)
            label_rect = self._label_item.boundingRect()
            self._label_item.setPos(
                mid.x() - label_rect.width() / 2,
                mid.y() - label_rect.height() / 2,
            )

    def set_highlighted(self, highlighted: bool):
        self._highlighted = highlighted
        self._dimmed = False
        color = QColor("#ffffff" if self.dark else "#000000") if highlighted else self.base_color
        self.setPen(QPen(color, 2.5 if highlighted else 1.5))
        if self._label_item:
            self._label_item.setDefaultTextColor(
                QColor("#ffffff" if self.dark else "#000000") if highlighted else (QColor("#cbd5e1") if self.dark else QColor("#334155"))
            )
        self.update()

    def set_dimmed(self, dimmed: bool):
        self._dimmed = dimmed
        self._highlighted = False
        color = QColor(self.base_color)
        if dimmed:
            color.setAlpha(40 if self.dark else 60)
        self.setPen(QPen(color, 1.0 if dimmed else 1.5))
        if self._label_item:
            label_color = QColor("#cbd5e1") if self.dark else QColor("#334155")
            if dimmed:
                label_color.setAlpha(40 if self.dark else 60)
            self._label_item.setDefaultTextColor(label_color)
        self.update()


class ConceptMapView(QGraphicsView):
    """Zoomable, pannable graphics view for the concept map."""

    node_clicked = pyqtSignal(str)  # node_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._zoom = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 3.0

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with scroll wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            if self._zoom < self._max_zoom:
                self.scale(factor, factor)
                self._zoom *= factor
        else:
            if self._zoom > self._min_zoom:
                self.scale(1 / factor, 1 / factor)
                self._zoom /= factor

    def mousePressEvent(self, event: QMouseEvent):
        """Handle node clicks for selection."""
        item = self.itemAt(event.pos())
        # Walk up to find the ConceptNode parent
        node = None
        while item is not None:
            if isinstance(item, ConceptNode):
                node = item
                break
            item = item.parentItem()

        if node and event.button() == Qt.MouseButton.LeftButton:
            self.node_clicked.emit(node.node_id)
        super().mousePressEvent(event)


class ConceptMapWidget(QWidget):
    """Complete concept map panel with controls and rendering."""

    def __init__(self, dark: bool = True, parent=None):
        super().__init__(parent)
        self.dark = dark
        self._nodes: dict[str, ConceptNode] = {}
        self._edges: list[ConceptEdge] = []
        self._selected_node_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Legend bar
        self._legend = QWidget()
        legend_layout = QHBoxLayout(self._legend)
        legend_layout.setContentsMargins(10, 6, 10, 6)
        legend_layout.setSpacing(12)

        palette = NODE_COLORS_DARK if dark else NODE_COLORS
        for node_type, colors in palette.items():
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {colors['bg']}; font-size: 14px;")
            legend_layout.addWidget(dot)
            label = QLabel(node_type.capitalize())
            label.setStyleSheet(
                f"color: {'#94a3b8' if dark else '#64748b'}; font-size: 11px;"
            )
            legend_layout.addWidget(label)

        legend_layout.addStretch()

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.setFixedHeight(24)
        self._reset_btn.setStyleSheet(
            f"background: {'#334155' if dark else '#e2e8f0'}; "
            f"color: {'#e2e8f0' if dark else '#475569'}; "
            "border: none; border-radius: 4px; padding: 2px 10px; font-size: 11px;"
        )
        self._reset_btn.clicked.connect(self._reset_view)
        legend_layout.addWidget(self._reset_btn)

        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setFixedSize(24, 24)
        self._zoom_out_btn.setStyleSheet(self._reset_btn.styleSheet())
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        legend_layout.addWidget(self._zoom_out_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedSize(24, 24)
        self._zoom_in_btn.setStyleSheet(self._reset_btn.styleSheet())
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        legend_layout.addWidget(self._zoom_in_btn)

        layout.addWidget(self._legend)

        # Graphics view
        self._scene = QGraphicsScene(self)
        bg_color = "#1e1e1e" if dark else "#f8fafc"
        self._scene.setBackgroundBrush(QBrush(QColor(bg_color)))

        self._view = ConceptMapView(self)
        self._view.setScene(self._scene)
        self._view.node_clicked.connect(self._on_node_clicked)
        layout.addWidget(self._view)

        # Empty state
        self._empty_label = QLabel("No concept map generated yet.\nSelect a session and click 'Generate Map'.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {'#64748b' if dark else '#94a3b8'}; font-size: 14px; padding: 40px;"
        )
        layout.addWidget(self._empty_label)
        self._view.setVisible(False)
        self._legend.setVisible(False)

        # Generating overlay
        self._generating_widget = QWidget(self)
        gen_layout = QVBoxLayout(self._generating_widget)
        gen_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._gen_title = QLabel("Generating Concept Map")
        self._gen_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gen_title.setFont(QFont("Helvetica Neue", 18, QFont.Weight.Bold))
        self._gen_title.setStyleSheet(f"color: {'#e2e8f0' if dark else '#1e293b'};")
        gen_layout.addWidget(self._gen_title)

        self._gen_dots = QLabel("")
        self._gen_dots.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gen_dots.setFont(QFont("Helvetica Neue", 24, QFont.Weight.Bold))
        self._gen_dots.setStyleSheet("color: #8b5cf6;")
        gen_layout.addWidget(self._gen_dots)

        self._gen_status = QLabel("Preparing transcript...")
        self._gen_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gen_status.setFont(QFont("Helvetica Neue", 12))
        self._gen_status.setStyleSheet(
            f"color: {'#94a3b8' if dark else '#64748b'}; padding: 10px;"
        )
        gen_layout.addWidget(self._gen_status)

        layout.addWidget(self._generating_widget)
        self._generating_widget.setVisible(False)

        # Dot animation timer
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_dots)
        self._dot_timer.setInterval(400)

    def load_graph(self, graph: dict[str, Any]):
        """Load a graph dict and render it. Hides the generating overlay."""
        self._generating_widget.setVisible(False)
        self._dot_timer.stop()
        self._scene.clear()
        self._nodes.clear()
        self._edges.clear()
        self._selected_node_id = None

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not nodes:
            self._view.setVisible(False)
            self._legend.setVisible(False)
            self._empty_label.setVisible(True)
            self._empty_label.setText("No concepts found in this session.")
            return

        self._empty_label.setVisible(False)
        self._view.setVisible(True)
        self._legend.setVisible(True)

        # Create nodes
        for n in nodes:
            node = ConceptNode(n, dark=self.dark)
            self._scene.addItem(node)
            self._nodes[n["id"]] = node

        # Create edges
        for e in edges:
            src = self._nodes.get(e.get("source"))
            tgt = self._nodes.get(e.get("target"))
            if src and tgt:
                edge = ConceptEdge(src, tgt, label=e.get("label", ""), dark=self.dark)
                self._scene.addItem(edge)
                self._edges.append(edge)

        # Run force-directed layout
        self._force_layout()

        # Fit view
        self._view.fitInView(self._scene.itemsBoundingRect().adjusted(-50, -50, 50, 50),
                             Qt.AspectRatioMode.KeepAspectRatio)
        self._view._zoom = 1.0

    def _force_layout(self):
        """Simple force-directed layout: nodes repel, edges attract."""
        nodes = list(self._nodes.values())
        if not nodes:
            return

        random.seed(42)  # Deterministic layout
        
        # Initialize random positions
        for node in nodes:
            node.setPos(random.uniform(-300, 300), random.uniform(-300, 300))

        # Build adjacency for attraction
        adjacency: dict[str, set[str]] = {n.node_id: set() for n in nodes}
        for edge in self._edges:
            adjacency[edge.source.node_id].add(edge.target.node_id)
            adjacency[edge.target.node_id].add(edge.source.node_id)

        # Iterate
        repulsion = 8000.0
        attraction = 0.01
        damping = 0.85
        velocities = {n.node_id: QPointF(0, 0) for n in nodes}

        for iteration in range(120):
            forces = {n.node_id: QPointF(0, 0) for n in nodes}

            # Repulsion (all pairs)
            for i, a in enumerate(nodes):
                for j, b in enumerate(nodes):
                    if i >= j:
                        continue
                    dx = b.x() - a.x()
                    dy = b.y() - a.y()
                    dist_sq = dx * dx + dy * dy + 1.0
                    force = repulsion / dist_sq
                    dist = math.sqrt(dist_sq)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[a.node_id] = QPointF(
                        forces[a.node_id].x() - fx,
                        forces[a.node_id].y() - fy,
                    )
                    forces[b.node_id] = QPointF(
                        forces[b.node_id].x() + fx,
                        forces[b.node_id].y() + fy,
                    )

            # Attraction (edges)
            for edge in self._edges:
                dx = edge.target.x() - edge.source.x()
                dy = edge.target.y() - edge.source.y()
                dist = math.sqrt(dx * dx + dy * dy) + 1.0
                force = attraction * dist
                fx = force * dx / dist
                fy = force * dy / dist
                forces[edge.source.node_id] = QPointF(
                    forces[edge.source.node_id].x() + fx,
                    forces[edge.source.node_id].y() + fy,
                )
                forces[edge.target.node_id] = QPointF(
                    forces[edge.target.node_id].x() - fx,
                    forces[edge.target.node_id].y() - fy,
                )

            # Apply forces with velocity damping
            max_move = 10.0
            for node in nodes:
                v = velocities[node.node_id]
                f = forces[node.node_id]
                vx = (v.x() + f.x()) * damping
                vy = (v.y() + f.y()) * damping
                # Clamp
                mag = math.sqrt(vx * vx + vy * vy) + 0.001
                if mag > max_move:
                    vx = vx / mag * max_move
                    vy = vy / mag * max_move
                velocities[node.node_id] = QPointF(vx, vy)
                node.setPos(node.x() + vx, node.y() + vy)

        # Update all edges after final positioning
        for edge in self._edges:
            edge.update_position()

    def _on_node_clicked(self, node_id: str):
        """Highlight clicked node and its neighbors, dim everything else."""
        if self._selected_node_id == node_id:
            # Deselect
            self._selected_node_id = None
            for n in self._nodes.values():
                n.set_highlighted(False)
                n.set_dimmed(False)
            for e in self._edges:
                e.set_highlighted(False)
                e.set_dimmed(False)
            return

        self._selected_node_id = node_id
        selected = self._nodes.get(node_id)
        if not selected:
            return

        # Find neighbors
        neighbor_ids = set()
        highlight_edges = []
        for edge in self._edges:
            if edge.source.node_id == node_id:
                neighbor_ids.add(edge.target.node_id)
                highlight_edges.append(edge)
            elif edge.target.node_id == node_id:
                neighbor_ids.add(edge.source.node_id)
                highlight_edges.append(edge)

        # Apply
        for nid, node in self._nodes.items():
            if nid == node_id:
                node.set_highlighted(True)
            elif nid in neighbor_ids:
                node.set_highlighted(False)
                node.set_dimmed(False)
            else:
                node.set_dimmed(True)

        for edge in self._edges:
            if edge in highlight_edges:
                edge.set_highlighted(True)
            else:
                edge.set_dimmed(True)

    def _reset_view(self):
        """Reset zoom and pan to fit all content."""
        self._selected_node_id = None
        for n in self._nodes.values():
            n.set_highlighted(False)
            n.set_dimmed(False)
        for e in self._edges:
            e.set_highlighted(False)
            e.set_dimmed(False)
        self._view.resetTransform()
        self._view._zoom = 1.0
        if self._nodes:
            self._view.fitInView(
                self._scene.itemsBoundingRect().adjusted(-50, -50, 50, 50),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def _zoom_in(self):
        """Zoom in on the concept map."""
        factor = 1.15
        if self._view._zoom < self._view._max_zoom:
            self._view.scale(factor, factor)
            self._view._zoom *= factor

    def _zoom_out(self):
        """Zoom out of the concept map."""
        factor = 1.15
        if self._view._zoom > self._view._min_zoom:
            self._view.scale(1 / factor, 1 / factor)
            self._view._zoom /= factor

    def clear_map(self):
        """Clear the map and show the empty state."""
        self._scene.clear()
        self._nodes.clear()
        self._edges.clear()
        self._selected_node_id = None
        self._view.setVisible(False)
        self._legend.setVisible(False)
        self._generating_widget.setVisible(False)
        self._dot_timer.stop()
        self._empty_label.setVisible(True)
        self._empty_label.setText("No concept map generated yet.\nSelect a session and click 'Generate Map'.")

    def show_generating(self, message: str = "Preparing transcript..."):
        """Show the generating overlay with animated progress."""
        self._empty_label.setVisible(False)
        self._view.setVisible(False)
        self._legend.setVisible(False)
        self._generating_widget.setVisible(True)
        self._gen_status.setText(message)
        self._dot_count = 0
        self._dot_timer.start()

    def update_status(self, message: str):
        """Update the generating status text."""
        self._gen_status.setText(message)

    def _animate_dots(self):
        self._dot_count = (self._dot_count + 1) % 4
        self._gen_dots.setText("\u25cf" * (self._dot_count + 1))

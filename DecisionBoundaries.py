from manim import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.datasets import make_moons

# ============================================================================
# WRAPPER CLASS TO MAINTAIN COMPATIBILITY WITH ORIGINAL ANIMATION CODE
# ============================================================================
class DecisionTreeClassifier:
    """Wrapper to make sklearn DecisionTreeClassifier compatible with original animation code"""
    def __init__(self, max_depth=None, criterion='gini', min_samples_leaf=1,
                 min_samples_split=2, lim_impurity=None, random_state=None):
        min_impurity_decrease = 0.0
        if lim_impurity is not None:
            min_impurity_decrease = lim_impurity

        self.sklearn_tree = SklearnDecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state
        )

    def fit(self, X, y):
        self.sklearn_tree.fit(X, y)
        return self

    def predict(self, X):
        return self.sklearn_tree.predict(X)

    def score(self, X, y):
        return self.sklearn_tree.score(X, y)

    def get_depth(self):
        return self.sklearn_tree.get_depth()

    def get_n_leaves(self):
        return self.sklearn_tree.get_n_leaves()

    def get_n_nodes(self):
        return self.sklearn_tree.tree_.node_count


# ============================================================================
# SCENE 5: Decision Boundary Evolution (Final)
# ============================================================================
class DecisionBoundaryEvolutionScene(Scene):
    def construct(self):
        # Top title
        title = Text("Decision Boundary Evolution", font_size=44, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.3)

        
        # Generate data
        np.random.seed(42)
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        self.X = X
        self.y = y
        self.X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 8 + 1

        # Left intro text (initial)
        self.intro_text = Text("Decision boundary for full complexity tree", font_size=24, color=YELLOW)
        self.intro_text.to_edge(LEFT).shift(UP * 0.5)
        self.play(Write(self.intro_text))
        self.wait(0.2)

        # Create right-side axes and show (persistent)
        self.create_coordinate_system()

        # Create stats box on lower-left (visible from start)
        self.create_stats_box()

        # Show data points (on axes)
        self.show_data_points()

        self.wait(0.6)

        # PART A: Show full complexity tree decision boundary
        tree_full = DecisionTreeClassifier(max_depth=None, criterion='gini', random_state=42)
        tree_full.fit(X, y)
        # Create boundary and stats for full tree
        # Note: we pass an empty status_text or short one but avoid "previous slide" comments anywhere
        self.update_boundary_and_stats(tree_full, status_text="Full complexity tree")
        self.wait(1.0)

        # -------------------------
        # TRANSITION TO EXPLANATION
        # -------------------------
        # Slide left-side elements (intro text + stats box & content) OUT to the LEFT
        left_group = VGroup(self.intro_text, self.stats_box, self.stats_title, self.stats_content)
        slide_left_out = left_group.animate.shift(LEFT * 8)

        # Slide right-side elements (axes, labels, data dots, boundary) OUT to the RIGHT
        rhs_group = VGroup(self.axes, self.axes_labels, self.data_dots)
        if hasattr(self, 'boundary_vgroup'):
            rhs_group.add(self.boundary_vgroup)
        slide_right_out = rhs_group.animate.shift(RIGHT * 12)

        # Perform both slides simultaneously
        self.play(slide_left_out, slide_right_out, run_time=1.0)
        self.wait(0.12)

        # Slide IN explanation box from bottom (center)
        self.show_explanation_box_from_bottom()

        # Wait for audience to read
        self.wait(1.4)

        # -------------------------
        # TRANSITION BACK (explanation down out -> content back in)
        # -------------------------
        # Slide explanation straight down out of frame (choice A)
        self.play(self.expl_box.animate.shift(DOWN * 6), FadeOut(self.expl_title), FadeOut(self.expl_items), run_time=0.9)
        # Remove expl objects from scene completely
        self.remove(self.expl_box, self.expl_title, self.expl_items)

        # Bring RHS content back from right
        self.play(rhs_group.animate.shift(LEFT * 12), run_time=1.0)

        # Bring stats/title back from left (slide back in)
        self.play(left_group.animate.shift(RIGHT * 8), run_time=1.0)

        # Ensure data points on top
        self.bring_to_front(self.data_dots)
        self.wait(0.2)

        # Now remove the intro text (we will replace left side with stopping criteria box)
        self.play(FadeOut(self.intro_text), run_time=0.4)

        # Create the new criteria box above the stats (fresh)
        self.create_criteria_box()

        # -------------------------
        # DEMONSTRATE STOPPING CRITERIA (exact sequences requested)
        # -------------------------

        # MAXIMUM DEPTH: 10, 5, 2
        for depth in [10, 5, 2]:
            self.update_criteria_box("MAXIMUM DEPTH", str(depth))
            tree = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
            tree.fit(X, y)
            self.update_boundary_and_stats(tree)  # no extraneous status text
            self.wait(0.9)

        # IMPURITY THRESHOLD: 0.005, 0.01, 0.05
        for lim_impurity in [0, 0.005, 0.01, 0.05]:
            self.update_criteria_box("IMPURITY THRESHOLD", f"{lim_impurity:.3f}")
            tree = DecisionTreeClassifier(max_depth=None, criterion='gini', lim_impurity=lim_impurity, random_state=42)
            tree.fit(X, y)
            self.update_boundary_and_stats(tree)
            self.wait(0.9)

        # MINIMUM SAMPLES: 3, 10, 50
        for min_samples in [3, 10, 50]:
            self.update_criteria_box("MINIMUM SAMPLES", str(min_samples))
            tree = DecisionTreeClassifier(max_depth=None, criterion='gini', min_samples_split=min_samples, random_state=42)
            tree.fit(X, y)
            self.update_boundary_and_stats(tree)
            self.wait(0.9)

        self.wait(1.0)

    # -------------------------
    # Helper functions
    # -------------------------
    def create_coordinate_system(self):
        """Create persistent coordinate system on the right side"""
        self.axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=5.5,
            y_length=5.5,
            axis_config={"include_tip": False, "font_size": 18}
        )
        self.axes.shift(RIGHT * 3 + DOWN * 0.5)
        axes_labels = self.axes.get_axis_labels(x_label="Feature 1", y_label="Feature 2")
        self.play(Create(self.axes), Write(axes_labels), run_time=0.9)
        self.axes_labels = axes_labels
        self.wait(0.08)

    def show_data_points(self):
        """Show data points on coordinate system"""
        self.data_dots = VGroup()
        for point, label in zip(self.X_normalized, self.y):
            dot = Dot(
                self.axes.c2p(point[0], point[1]),
                color=PURE_BLUE if label == 0 else PURE_RED,
                radius=0.07,
                stroke_width=1,
                stroke_color=WHITE,
                fill_opacity=1.0
            )
            self.data_dots.add(dot)

        self.data_dots.set_z_index(10)
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in self.data_dots], lag_ratio=0.02), run_time=1.4)
        self.bring_to_front(self.data_dots)
        self.wait(0.06)

    def create_stats_box(self):
        """Create statistics box in lower left corner"""
        self.stats_box = Rectangle(width=3.5, height=2, color=WHITE, stroke_width=2, fill_opacity=0.05)
        self.stats_box.to_corner(DL).shift(UP * 0.3 + RIGHT * 0.3)

        stats_title = Text("Tree Statistics", font_size=20, weight=BOLD, color=GREEN)
        stats_title.next_to(self.stats_box.get_top(), DOWN, buff=0.15)

        self.stats_content = VGroup(
            Text("Accuracy: --", font_size=18),
            Text("Tree Depth: --", font_size=18),
            Text("Leaf Nodes: --", font_size=18),
            Text("Total Nodes: --", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        self.stats_content.next_to(stats_title, DOWN, buff=0.2)

        self.play(Create(self.stats_box), Write(stats_title))
        self.play(Write(self.stats_content), run_time=0.7)
        self.stats_title = stats_title
        self.wait(0.08)

    # -------------------------
    # Explanation box: slide up from bottom and slide down out
    # -------------------------
    def show_explanation_box_from_bottom(self):
        """Create and show a styled explanation box sliding up from bottom"""
        expl = RoundedRectangle(width=8, height=4, corner_radius=0.35, stroke_width=4, color=TEAL, fill_opacity=0.06)
        # position below screen initially
        expl.to_edge(DOWN)
        expl.shift(DOWN * 3)  # off-screen start
        expl_title = Text("Stopping Criteria", font_size=32, weight=BOLD, color=TEAL)
        expl_title.move_to(expl.get_top() + DOWN * 0.6)

        bullet_items = VGroup(
            Text("• Maximum Depth", font_size=24),
            Text("• Threshold for impurity", font_size=24),
            Text("• Minimal number of samples", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        bullet_items.next_to(expl_title, DOWN, buff=0.45)

        # Bring expl into center by sliding up
        self.add(expl, expl_title, bullet_items)
        self.play(expl.animate.shift(UP * 3), expl_title.animate.shift(UP * 3), bullet_items.animate.shift(UP * 3), run_time=1.0)
        self.expl_box = expl
        self.expl_title = expl_title
        self.expl_items = bullet_items
        self.wait(0.08)

    # -------------------------
    # Criteria box (left) creation and update
    # -------------------------
    def create_criteria_box(self):
        """Create the stopping-criteria box above stats_box (fresh)"""
        self.criteria_box = RoundedRectangle(width=3.5, height=2.5, corner_radius=0.12, stroke_width=3, color=YELLOW, fill_opacity=0.03)
        self.criteria_box.next_to(self.stats_box, UP, buff=0.3)
        self.criteria_title = Text("Criterion", font_size=22, weight=BOLD, color=YELLOW)
        self.criteria_title.next_to(self.criteria_box.get_top(), DOWN, buff=0.15)
        self.criteria_value = Text("--", font_size=60, weight=BOLD, color=WHITE)
        self.criteria_value.move_to(self.criteria_box.get_center() + DOWN * 0.08)
        self.play(Create(self.criteria_box), Write(self.criteria_title))
        self.play(Write(self.criteria_value), run_time=0.6)
        self.wait(0.06)

    def update_criteria_box(self, title, value_text):
        """Update title and big number in the criteria box"""
        new_title = Text(title, font_size=22, weight=BOLD, color=YELLOW)
        new_title.next_to(self.criteria_box.get_top(), DOWN, buff=0.15)
        new_value = Text(value_text, font_size=60, weight=BOLD, color=WHITE)
        new_value.move_to(self.criteria_box.get_center() + DOWN * 0.08)
        self.play(Transform(self.criteria_title, new_title), Transform(self.criteria_value, new_value), run_time=0.6)
        self.wait(0.06)

    # -------------------------
    # Update boundary and stats (cleaned — minimal on-screen status)
    # -------------------------
    def update_boundary_and_stats(self, tree, status_text=""):
        """Update decision boundary by creating or morphing blocks, and update stats."""
        h = 0.05
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.3, self.X[:, 1].max() + 0.3

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # If first time, create blocks
        if not hasattr(self, 'boundary_blocks'):
            self.boundary_blocks = {}
            self.boundary_vgroup = VGroup()
            xx_norm = (xx - self.X[:, 0].min()) / (self.X[:, 0].max() - self.X[:, 0].min()) * 8 + 1
            yy_norm = (yy - self.X[:, 1].min()) / (self.X[:, 1].max() - self.X[:, 1].min()) * 8 + 1

            for i in range(0, len(xx) - 1, 1):
                for j in range(0, len(yy[0]) - 1, 1):
                    if Z[i, j] == 0:
                        color = BLUE_B  # Full blue intensity
                    else:
                        color = RED_C  # Full red intensity
                
                    square = Square(
                    side_length=h * 2 * 5.5/10,  # Match BoostingBoundary axes scaling
                    color=color,
                    fill_opacity=0.8,  # Fixed high opacity for hard predictions
                    stroke_width=0
                    )
                    square.move_to(self.axes.c2p(xx_norm[i, j], yy_norm[i, j]))
                    square.set_z_index(1)
                    self.boundary_blocks[(i, j)] = {'square': square, 'current_class': Z[i, j]}
                    self.boundary_vgroup.add(square)

            self.play(FadeIn(self.boundary_vgroup), run_time=1.0)
            self.bring_to_front(self.data_dots)

        else:
            # Find changes and animate
            changed = []
            for i in range(0, len(xx) - 1, 1):
                for j in range(0, len(yy[0]) - 1, 1):
                    key = (i, j)
                    if key in self.boundary_blocks:
                        info = self.boundary_blocks[key]
                        old = info['current_class']
                        new = Z[i, j]
                        if old != new:
                            color = BLUE_B if new == 0 else RED_C
                            changed.append({'square': info['square'], 'new_color': color, 'new_class': new})
         
            if changed:
                # flash white then change to new color
                self.play(*[c['square'].animate.set_fill(WHITE, opacity=0.9) for c in changed], run_time=0.25)
                self.play(*[c['square'].animate.set_fill(c['new_color'], opacity=0.8) for c in changed], run_time=0.8)
                # update stored classes
                for c in changed:
                    for key, block_info in self.boundary_blocks.items():
                        if block_info['square'] == c['square']:
                            block_info['current_class'] = c['new_class']
                            break
                self.bring_to_front(self.data_dots)
            else:
                self.wait(0.2)

        # Update stats display (accuracy, depth, leaves, nodes)
        accuracy = tree.score(self.X, self.y)
        depth = tree.get_depth()
        n_leaves = tree.get_n_leaves()
        n_nodes = tree.get_n_nodes()

        new_stats = VGroup(
            Text(f"Accuracy: {accuracy:.1%}", font_size=18, color=GREEN),
            Text(f"Tree Depth: {depth}", font_size=18),
            Text(f"Leaf Nodes: {n_leaves}", font_size=18),
            Text(f"Total Nodes: {n_nodes}", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        new_stats.move_to(self.stats_content.get_center())

        self.play(Transform(self.stats_content, new_stats), run_time=0.45)
        # Keep data points on top
        self.bring_to_front(self.data_dots)

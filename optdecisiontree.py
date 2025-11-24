from manim import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

class OptimizedDecisionTreeScene(Scene):
    def construct(self):
        # Title
        title = Text("Finding the Optimal Decision Tree Split", font_size=40, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title
        self.wait(0.5)
        
        # Generate data - same as previous scenes
        np.random.seed(42)
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        self.X = X
        self.y = y
        
        # Normalize data for display (0-8 range)
        self.X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 8 + 1
        
        # ============================================================
        # PART 1: Display data with decision boundary visualization
        # ============================================================
        self.create_data_visualization()
        
        # ============================================================
        # PART 2: Show impurity measures and calculation boxes
        # ============================================================
        self.create_impurity_box()
        self.create_calculation_box()
        
        # Calculate and display initial impurity
        self.calculate_initial_impurity()
        
        # ============================================================
        # PART 3: Animate VERTICAL split exploration (Feature 1) FIRST
        # ============================================================
        self.explore_vertical_splits_detailed()
        
        # ============================================================
        # PART 4: Animate HORIZONTAL split exploration (Feature 2) SECOND
        # ============================================================
        self.explore_horizontal_splits_simple()
        
        # ============================================================
        # PART 5: Compare and show winner with decision boundary
        # ============================================================
        self.show_optimal_split_with_boundary()
        
        # ============================================================
        # PART 6: Grow full complexity tree with better spacing
        # ============================================================
        self.grow_full_tree_exact()
        
        # ============================================================
        # PART 7: Show overfitting alert
        # ============================================================
        self.overfitting_alert()

        self.wait(3)
    
    def create_data_visualization(self):
        """Create coordinate system and data points on right side"""
        # Create axes
        self.axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=5.5,
            y_length=5.5,
            axis_config={"include_tip": False, "font_size": 18}
        )
        self.axes.shift(RIGHT * 3.5 + DOWN * 0.5)
        
        # Create axis labels OUTSIDE (below x-axis, left of y-axis)
        x_label = Text("Feature 1", font_size=18)
        x_label.next_to(self.axes.x_axis, DOWN, buff=0.3)
        
        y_label = Text("Feature 2", font_size=18)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(self.axes.y_axis, LEFT, buff=0.3)
        
        self.axes_labels = VGroup(x_label, y_label)
        
        self.play(Create(self.axes), Write(self.axes_labels), run_time=1.5)
        self.wait(0.5)
        
        # Create data points with PURE_BLUE and PURE_RED
        self.data_dots = VGroup()
        for point, label in zip(self.X_normalized, self.y):
            dot = Dot(
                self.axes.c2p(point[0], point[1]),
                color=PURE_BLUE if label == 0 else PURE_RED,
                radius=0.07,
                stroke_width=1.5,
                stroke_color=WHITE,
                fill_opacity=1.0
            )
            dot.set_z_index(10)  # Ensure points are on top
            self.data_dots.add(dot)
        
        self.play(
            LaggedStart(*[GrowFromCenter(dot) for dot in self.data_dots], 
                       lag_ratio=0.02),
            run_time=2
        )
        self.bring_to_front(self.data_dots)
        self.wait(0.5)
    
    def create_impurity_box(self):
        """Create impurity measures box on left side"""
        # Create formula box
        self.formula_box = Rectangle(
            width=5, height=3, 
            color=WHITE, 
            stroke_width=2, 
            fill_opacity=0.05
        )
        self.formula_box.to_edge(LEFT).shift(RIGHT * 0.3 + UP * 1.1)
        
        # Title
        formula_title = Text(
            "Impurity Measures", 
            font_size=20, 
            weight=BOLD, 
            color=RED
        )
        formula_title.next_to(self.formula_box.get_top(), DOWN, buff=0.15)
        
        # Gini Index formula
        gini_label = Text("Gini Index:", font_size=16, color=YELLOW, weight=BOLD)
        gini_formula = MathTex(
            r"G(R) = \sum_{k=1}^{K} \hat{p}_k(1 - \hat{p}_k)",
            font_size=24
        )
        gini_group = VGroup(gini_label, gini_formula).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        
        # Information Gain formula
        ig_label = Text("Information Gain:", font_size=16, color=GREEN, weight=BOLD)
        self.ig_formula_template = MathTex(
            r"IG = G_{parent} - (n_L/n) \cdot G_L - (n_R/n) \cdot G_R",
            font_size=24
        )
        ig_group = VGroup(ig_label, self.ig_formula_template).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        
        # Arrange formulas
        all_formulas = VGroup(gini_group, ig_group)
        all_formulas.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        all_formulas.move_to(self.formula_box.get_center())
        
        self.play(Create(self.formula_box), Write(formula_title))
        self.play(Write(gini_group), run_time=1)
        self.play(Write(ig_group), run_time=1)
        
        self.formula_title = formula_title
        self.formulas = all_formulas
        self.wait(0.5)
    
    def create_calculation_box(self):
        """Create calculation box below impurity box"""
        self.calc_box = Rectangle(
            width=5, height=2.5, 
            color=GREEN, 
            stroke_width=2, 
            fill_opacity=0.05
        )
        self.calc_box.next_to(self.formula_box, DOWN, buff=0.3)
        
        calc_title = Text(
            "Calculation / Values", 
            font_size=20, 
            weight=BOLD, 
            color=GREEN
        )
        calc_title.next_to(self.calc_box.get_top(), DOWN, buff=0.15)
        
        self.play(Create(self.calc_box), Write(calc_title))
        self.calc_title = calc_title
        self.calc_content = None
        self.wait(0.5)
    
    def calculate_initial_impurity(self):
        """Calculate and display initial data impurity as G(R) = 0.5"""
        n_total = len(self.y)
        n_class0 = np.sum(self.y == 0)
        n_class1 = np.sum(self.y == 1)
        
        p0 = n_class0 / n_total
        p1 = n_class1 / n_total
        gini_initial = 2 * p0 * p1
        
        # Display G(R) = 0.5 centered in calculation box
        self.gini_display = MathTex(r"G(R) = 0.5", font_size=24, color=WHITE)
        self.gini_display.next_to(self.calc_title, DOWN, buff=0.3)
        self.gini_display.move_to(self.calc_box.get_center() + UP * 0.5)
        
        self.play(Write(self.gini_display))
        
        self.initial_gini = gini_initial
        self.wait(1)
    
    def explore_vertical_splits_detailed(self):
        """Animate VERTICAL split exploration for Feature 1 with detailed calculation - NOW FIRST"""
        # Create vertical split line IN THE MIDDLE
        x_min = self.X_normalized[:, 0].min()
        x_max = self.X_normalized[:, 0].max()
        x_start = (x_min + x_max) / 2  # Start in the middle
        
        self.split_line = Line(
            self.axes.c2p(x_start, 0),
            self.axes.c2p(x_start, 10),
            color=ORANGE,
            stroke_width=4
        )
        self.split_line.set_z_index(5)
        
        # "Threshold" label ABOVE the line (top)
        threshold_label = Text(
            "Threshold",
            font_size=16,
            color=ORANGE
        )
        threshold_label.next_to(self.split_line.get_top(), UP, buff=0.1)
        
        # Value label BELOW x-axis
        threshold_value = Text(
            f"{x_start:.2f}",
            font_size=14,
            color=ORANGE
        )
        threshold_value.next_to(self.axes.c2p(x_start, 0), DOWN, buff=0.1)
        
        self.play(
            Create(self.split_line), 
            Write(threshold_label),
            Write(threshold_value)
        )
        self.threshold_label = threshold_label
        self.threshold_value = threshold_value
        self.wait(0.5)
        
        # Calculate IG for starting position
        initial_ig = self.calculate_ig_for_threshold(x_start, axis=0)
        
        # DETAILED IG CALCULATION ANIMATION - IN CALC BOX ONLY
        # Display IG formula
        ig_formula_display = MathTex(
            r"IG_1 = G_{parent} - (n_L/n) \cdot G_L - (n_R/n) \cdot G_R",
            font_size=20,
            color=ORANGE
        )
        ig_formula_display.next_to(self.gini_display, DOWN, buff=0.3)
        
        self.play(Write(ig_formula_display), run_time=1)
        self.wait(0.5)
        
        # Calculate actual values for current split
        left_mask = self.X_normalized[:, 0] <= x_start
        right_mask = ~left_mask
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(self.y)
        
        # Calculate Gini for each split
        left_labels = self.y[left_mask]
        right_labels = self.y[right_mask]
        
        p_left_0 = np.sum(left_labels == 0) / n_left if n_left > 0 else 0
        p_left_1 = 1 - p_left_0
        gini_left = 2 * p_left_0 * p_left_1
        
        p_right_0 = np.sum(right_labels == 0) / n_right if n_right > 0 else 0
        p_right_1 = 1 - p_right_0
        gini_right = 2 * p_right_0 * p_right_1
        
        # Fill in the formula with actual values (ORANGE color)
        ig_formula_filled = MathTex(
            f"IG_1 = 0.5 - ({n_left}/200) \cdot {gini_left:.3f} - ({n_right}/200) \cdot {gini_right:.3f}",
            font_size=18,
            color=ORANGE
        )
        ig_formula_filled.move_to(ig_formula_display.get_center())
        
        # Transform to filled formula
        self.play(Transform(ig_formula_display, ig_formula_filled), run_time=1)
        self.wait(0.5)
        
        # Transform to BOLD BIG value (ORANGE) - use larger font instead of weight
        ig_value_bold = MathTex(
            f"\\mathbf{{IG_1 = {initial_ig:.4f}}}",
            font_size=36,
            color=ORANGE
        )
        ig_value_bold.move_to(ig_formula_display.get_center())
        
        self.play(Transform(ig_formula_display, ig_value_bold), run_time=1)
        self.wait(0.5)
        
        # Track best split
        self.best_vertical_ig = initial_ig
        self.best_vertical_threshold = x_start
        
        # ANIMATE threshold changes - UPDATE THE SAME VALUE IN CALC BOX
        n_steps = 90
        x_values_left = np.linspace(x_start, x_min + 0.3, n_steps//2)
        x_values_right = np.linspace(x_min + 0.3, x_max - 0.3, n_steps)
        
        # Move left
        for x_val in x_values_left:
            ig = self.calculate_ig_for_threshold(x_val, axis=0)
            if ig > self.best_vertical_ig:
                self.best_vertical_ig = ig
                self.best_vertical_threshold = x_val
            
            # Update the value in place
            new_ig_value = MathTex(
                f"\\mathbf{{IG_1 = {ig:.4f}}}",
                font_size=36,
                color=ORANGE
            )
            new_ig_value.move_to(ig_formula_display.get_center())
            
            self.play(
                self.split_line.animate.move_to(self.axes.c2p(x_val, 5)),
                self.threshold_label.animate.next_to(self.axes.c2p(x_val, 10), UP, buff=0.1),
                self.threshold_value.animate.become(
                    Text(f"{x_val:.2f}", font_size=14, color=ORANGE)
                ).next_to(self.axes.c2p(x_val, 0), DOWN, buff=0.1),
                Transform(ig_formula_display, new_ig_value),
                run_time=0.04
            )
        
        # Move right
        for x_val in x_values_right:
            ig = self.calculate_ig_for_threshold(x_val, axis=0)
            if ig > self.best_vertical_ig:
                self.best_vertical_ig = ig
                self.best_vertical_threshold = x_val
            
            # Update the value in place
            new_ig_value = MathTex(
                f"\\mathbf{{IG_1 = {ig:.4f}}}",
                font_size=36,
                color=ORANGE
            )
            new_ig_value.move_to(ig_formula_display.get_center())
            
            self.play(
                self.split_line.animate.move_to(self.axes.c2p(x_val, 5)),
                self.threshold_label.animate.next_to(self.axes.c2p(x_val, 10), UP, buff=0.1),
                self.threshold_value.animate.become(
                    Text(f"{x_val:.2f}", font_size=14, color=ORANGE)
                ).next_to(self.axes.c2p(x_val, 0), DOWN, buff=0.1),
                Transform(ig_formula_display, new_ig_value),
                run_time=0.04
            )
        
        # Move to best position with GREEN indicator
        best_ig_bold = MathTex(
            f"\\mathbf{{IG_1 = {self.best_vertical_ig:.4f}}}",
            font_size=36,
            color=GREEN
        )
        best_ig_bold.move_to(ig_formula_display.get_center())
        
        self.play(
            self.split_line.animate.move_to(self.axes.c2p(self.best_vertical_threshold, 5)),
            self.threshold_label.animate.next_to(self.axes.c2p(self.best_vertical_threshold, 10), UP, buff=0.1),
            self.threshold_value.animate.become(
                Text(f"{self.best_vertical_threshold:.2f}", font_size=14, color=GREEN, weight=BOLD)
            ).next_to(self.axes.c2p(self.best_vertical_threshold, 0), DOWN, buff=0.1),
            Transform(ig_formula_display, best_ig_bold),
            run_time=1
        )
        self.wait(0.5)
        
        # Add star and shrink to normal size
        ig_star_1 = MathTex(
            f"IG^*_1 = {self.best_vertical_ig:.4f}",
            font_size=20,
            color=ORANGE
        )
        ig_star_1.move_to(ig_formula_display.get_center())
        
        self.play(Transform(ig_formula_display, ig_star_1), run_time=0.8)
        self.ig_star_1_display = ig_formula_display
        self.wait(0.5)
        
        # Clean up
        self.play(
            FadeOut(self.split_line),
            FadeOut(self.threshold_label),
            FadeOut(self.threshold_value),
            run_time=0.5
        )
        self.wait(0.5)
    
    def explore_horizontal_splits_simple(self):
        """Animate HORIZONTAL split exploration for Feature 2 - simpler version (NO detailed calculation)"""
        # Create horizontal split line in the middle
        y_min = self.X_normalized[:, 1].min()
        y_max = self.X_normalized[:, 1].max()
        y_start = (y_min + y_max) / 2
        
        self.split_line = Line(
            self.axes.c2p(0, y_start),
            self.axes.c2p(10, y_start),
            color=YELLOW,
            stroke_width=4
        )
        self.split_line.set_z_index(5)
        
        # "Threshold" label on TOP RIGHT of line
        threshold_label = Text(
            "Threshold",
            font_size=16,
            color=YELLOW
        )
        threshold_label.next_to(self.split_line.get_end(), UP, buff=0.1)
        
        # Value label on LEFT by y-axis
        threshold_value = Text(
            f"{y_start:.2f}",
            font_size=14,
            color=YELLOW
        )
        threshold_value.next_to(self.axes.c2p(0, y_start), LEFT, buff=0.1)
        
        self.play(
            Create(self.split_line),
            Write(threshold_label),
            Write(threshold_value)
        )
        self.threshold_label_f2 = threshold_label
        self.threshold_value_f2 = threshold_value
        self.wait(0.5)
        
        # Calculate initial IG
        initial_ig = self.calculate_ig_for_threshold(y_start, axis=1)
        
        # Display IG_2 BOLD in calc box
        ig_value_bold = MathTex(
            f"\\mathbf{{IG_2 = {initial_ig:.4f}}}",
            font_size=36,
            color=YELLOW
        )
        ig_value_bold.next_to(self.ig_star_1_display, DOWN, buff=0.3)
        
        self.play(Write(ig_value_bold), run_time=0.5)
        ig_formula_display = ig_value_bold
        
        # Track best split
        self.best_horizontal_ig = initial_ig
        self.best_horizontal_threshold = y_start
        
        # ANIMATE threshold changes
        n_steps = 90
        y_values_down = np.linspace(y_start, y_min + 0.3, n_steps//3)
        y_values_up = np.linspace(y_min + 0.3, y_max - 0.3, n_steps*2//3)
        
        # Move down
        for y_val in y_values_down:
            ig = self.calculate_ig_for_threshold(y_val, axis=1)
            if ig > self.best_horizontal_ig:
                self.best_horizontal_ig = ig
                self.best_horizontal_threshold = y_val
            
            # Update value
            new_ig_value = MathTex(
                f"\\mathbf{{IG_2 = {ig:.4f}}}",
                font_size=36,
                color=YELLOW
            )
            new_ig_value.move_to(ig_formula_display.get_center())
            
            self.play(
                self.split_line.animate.move_to(self.axes.c2p(5, y_val)),
                self.threshold_label_f2.animate.next_to(self.axes.c2p(10, y_val), UP, buff=0.1),
                self.threshold_value_f2.animate.become(
                    Text(f"{y_val:.2f}", font_size=14, color=YELLOW)
                ).next_to(self.axes.c2p(0, y_val), LEFT, buff=0.1),
                Transform(ig_formula_display, new_ig_value),
                run_time=0.04
            )
        
        # Move up
        for y_val in y_values_up:
            ig = self.calculate_ig_for_threshold(y_val, axis=1)
            if ig > self.best_horizontal_ig:
                self.best_horizontal_ig = ig
                self.best_horizontal_threshold = y_val
            
            # Update value
            new_ig_value = MathTex(
                f"\\mathbf{{IG_2 = {ig:.4f}}}",
                font_size=36,
                color=YELLOW
            )
            new_ig_value.move_to(ig_formula_display.get_center())
            
            self.play(
                self.split_line.animate.move_to(self.axes.c2p(5, y_val)),
                self.threshold_label_f2.animate.next_to(self.axes.c2p(10, y_val), UP, buff=0.1),
                self.threshold_value_f2.animate.become(
                    Text(f"{y_val:.2f}", font_size=14, color=YELLOW)
                ).next_to(self.axes.c2p(0, y_val), LEFT, buff=0.1),
                Transform(ig_formula_display, new_ig_value),
                run_time=0.04
            )
        
        # Move to best position with GREEN
        best_ig_bold = MathTex(
            f"\\mathbf{{IG_2 = {self.best_horizontal_ig:.4f}}}",
            font_size=36,
            color=GREEN
        )
        best_ig_bold.move_to(ig_formula_display.get_center())
        
        self.play(
            self.split_line.animate.move_to(self.axes.c2p(5, self.best_horizontal_threshold)),
            self.threshold_label_f2.animate.next_to(self.axes.c2p(10, self.best_horizontal_threshold), UP, buff=0.1),
            self.threshold_value_f2.animate.become(
                Text(f"{self.best_horizontal_threshold:.2f}", font_size=14, color=GREEN, weight=BOLD)
            ).next_to(self.axes.c2p(0, self.best_horizontal_threshold), LEFT, buff=0.1),
            Transform(ig_formula_display, best_ig_bold),
            run_time=1
        )
        self.wait(0.5)
        
        # Add star and shrink
        ig_star_2 = MathTex(
            f"IG^*_2 = {self.best_horizontal_ig:.4f}",
            font_size=20,
            color=YELLOW
        )
        ig_star_2.move_to(ig_formula_display.get_center())
        
        self.play(Transform(ig_formula_display, ig_star_2), run_time=0.8)
        self.ig_star_2_display = ig_formula_display
        self.wait(0.5)
        
        # Clean up Feature 2 threshold labels
        self.play(
            FadeOut(self.split_line),
            FadeOut(self.threshold_label_f2),
            FadeOut(self.threshold_value_f2),
            run_time=0.5
        )
        
        # Compare and show best split - CENTERED AT BOTTOM OF CALC BOX
        if self.best_vertical_ig > self.best_horizontal_ig:
            winner = Text("Best Split: Feature 1!", font_size=18, color=GREEN, weight=BOLD)
            self.optimal_feature = 0
            self.optimal_threshold = self.best_vertical_threshold
        else:
            winner = Text("Best Split: Feature 2!", font_size=18, color=GREEN, weight=BOLD)
            self.optimal_feature = 1
            self.optimal_threshold = self.best_horizontal_threshold
        
        winner.next_to(self.calc_box.get_bottom(), UP, buff=0.15)
        self.play(Write(winner))
        self.winner_text = winner
        self.wait(2)
    
    def calculate_ig_for_threshold(self, threshold, axis):
        """Calculate information gain for a given threshold on specified axis"""
        # Split data based on threshold
        if axis == 0:  # Feature 1 (vertical split)
            left_mask = self.X_normalized[:, 0] <= threshold
        else:  # Feature 2 (horizontal split)
            left_mask = self.X_normalized[:, 1] <= threshold
        
        right_mask = ~left_mask
        
        # Count samples
        n_total = len(self.y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Calculate Gini for left split
        left_labels = self.y[left_mask]
        p_left_0 = np.sum(left_labels == 0) / n_left if n_left > 0 else 0
        p_left_1 = np.sum(left_labels == 1) / n_left if n_left > 0 else 0
        gini_left = 2 * p_left_0 * p_left_1
        
        # Calculate Gini for right split
        right_labels = self.y[right_mask]
        p_right_0 = np.sum(right_labels == 0) / n_right if n_right > 0 else 0
        p_right_1 = np.sum(right_labels == 1) / n_right if n_right > 0 else 0
        gini_right = 2 * p_right_0 * p_right_1
        
        # Calculate information gain
        ig = self.initial_gini - (n_left/n_total * gini_left + n_right/n_total * gini_right)
        
        return ig
    
    def show_optimal_split_with_boundary(self):
        """Show decision stump on left and decision boundary on right"""
        # Fade out the formula and calculation boxes
        self.play(
            FadeOut(self.formula_box),
            FadeOut(self.formula_title),
            FadeOut(self.formulas),
            FadeOut(self.calc_box),
            FadeOut(self.calc_title),
            FadeOut(self.gini_display),
            FadeOut(self.ig_star_1_display),
            FadeOut(self.ig_star_2_display),
            FadeOut(self.winner_text),
            run_time=1
        )
        
        # Fit decision stump
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(self.X, self.y)
        
        # Create tree stump on LEFT side
        self.create_tree_stump_left(stump)
        
        # Create decision boundary on RIGHT side with consistent colors
        self.create_decision_boundary(stump)
        
        self.wait(3)
        
        self.stump = stump
    
    def create_tree_stump_left(self, tree):
        """Create tree stump visualization on left side"""
        tree_structure = tree.tree_
        
        left_center = LEFT * 4 + UP * 0.5
        
        # Root node
        root_feature = tree_structure.feature[0]
        root_threshold = tree_structure.threshold[0]
        root_impurity = tree_structure.impurity[0]
        root_samples = tree_structure.n_node_samples[0]
        
        root_shape = RoundedRectangle(
            width=2.0,
            height=1.0,
            corner_radius=0.1,
            color=ORANGE,
            fill_opacity=0.4,
            stroke_width=3
        )
        root_shape.move_to(left_center + UP * 1.5)
        
        root_label = VGroup(
            Text(f"X{root_feature} ≤ {root_threshold:.2f}", 
                 font_size=16, weight=BOLD),
            Text(f"gini = {root_impurity:.3f}", font_size=14),
            Text(f"samples = {root_samples}", font_size=14)
        ).arrange(DOWN, buff=0.05)
        root_label.move_to(root_shape.get_center())
        
        # Left child
        left_value = np.argmax(tree_structure.value[1])
        left_impurity = tree_structure.impurity[1]
        left_samples = tree_structure.n_node_samples[1]
        
        left_shape = Circle(
            radius=0.5,
            color=BLUE if left_value == 0 else RED,
            fill_opacity=0.7,
            stroke_width=3,
            stroke_color=WHITE
        )
        left_shape.move_to(left_center + LEFT * 1.5 + DOWN * 0.5)
        
        left_label = VGroup(
            Text(f"Class {left_value}", font_size=14, weight=BOLD, color=WHITE),
            Text(f"gini = {left_impurity:.3f}", font_size=12, color=WHITE),
            Text(f"n = {left_samples}", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.02)
        left_label.move_to(left_shape.get_center())
        
        # Right child
        right_value = np.argmax(tree_structure.value[2])
        right_impurity = tree_structure.impurity[2]
        right_samples = tree_structure.n_node_samples[2]
        
        right_shape = Circle(
            radius=0.5,
            color=BLUE if right_value == 0 else RED,
            fill_opacity=0.7,
            stroke_width=3,
            stroke_color=WHITE
        )
        right_shape.move_to(left_center + RIGHT * 1.5 + DOWN * 0.5)
        
        right_label = VGroup(
            Text(f"Class {right_value}", font_size=14, weight=BOLD, color=WHITE),
            Text(f"gini = {right_impurity:.3f}", font_size=12, color=WHITE),
            Text(f"n = {right_samples}", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.02)
        right_label.move_to(right_shape.get_center())
        
        # Edges
        left_edge = Line(
            root_shape.get_bottom(),
            left_shape.get_top(),
            color=BLUE,
            stroke_width=3
        )
        
        right_edge = Line(
            root_shape.get_bottom(),
            right_shape.get_top(),
            color=RED,
            stroke_width=3
        )
        
        # Animate
        self.play(
            Create(root_shape),
            Write(root_label),
            run_time=1
        )
        
        self.play(
            Create(left_edge),
            Create(right_edge),
            run_time=0.8
        )
        
        self.play(
            FadeIn(left_shape, scale=0.5),
            Write(left_label),
            FadeIn(right_shape, scale=0.5),
            Write(right_label),
            run_time=1
        )
        
        self.tree_stump_group = VGroup(
            root_shape, root_label,
            left_shape, left_label,
            right_shape, right_label,
            left_edge, right_edge
        )
    
    def create_decision_boundary(self, tree):
        """Create decision boundary matching BoostingBoundary style exactly"""
        h = 0.05  # Same as BoostingBoundary
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.3, self.X[:, 1].max() + 0.3  # Changed to -0.5/+0.5
        
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )
        
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Normalize for display - SAME as BoostingBoundary
        xx_norm = (xx - self.X[:, 0].min()) / (self.X[:, 0].max() - self.X[:, 0].min()) * 8 + 1
        yy_norm = (yy - self.X[:, 1].min()) / (self.X[:, 1].max() - self.X[:, 1].min()) * 8 + 1
        
        # Create boundary blocks with BoostingBoundary style
        boundary_blocks = VGroup()
        for i in range(0, len(xx) - 1, 1):  # Skip by 2 like BoostingBoundary
            for j in range(0, len(yy[0]) - 1, 1):
                # Use interpolate_color for consistent look (even with hard predictions)
                if Z[i, j] == 0:
                    color = interpolate_color(BLUE_C, BLUE_B, 1.0)  # Full blue intensity
                else:
                    color = interpolate_color(RED_B, RED_C, 1.0)  # Full red intensity
                
                square = Square(
                    side_length=h * 2 * 5.5/10,  # Match BoostingBoundary axes scaling
                    color=color,
                    fill_opacity=0.8,  # Fixed high opacity for hard predictions
                    stroke_width=0
                )
                square.move_to(self.axes.c2p(xx_norm[i, j], yy_norm[i, j]))
                square.set_z_index(1)
                boundary_blocks.add(square)
        
        self.play(FadeIn(boundary_blocks), run_time=1)
        
        # Ensure data points are on top
        self.bring_to_front(self.data_dots)
        
        self.boundary_blocks = boundary_blocks
    
    def grow_full_tree_exact(self):
        """Grow full complexity tree with BETTER SPACING for long chains"""
        # Fade everything except axes labels
        self.play(
            FadeOut(self.boundary_blocks),
            FadeOut(self.axes),
            FadeOut(self.data_dots),
            FadeOut(self.axes_labels),
            self.tree_stump_group.animate.scale(0.6).move_to(UP * 1.5),
            run_time=1.5
        )
        
        # Add "Continue tree growing" text
        continue_text = Text(
            "Continue Tree Growing...",
            font_size=28,
            color=GREEN,
            weight=BOLD
        )
        continue_text.next_to(self.tree_stump_group, DOWN, buff=0.5)
        self.play(Write(continue_text))
        self.wait(1)
        
        # Fade out stump and text
        self.play(
            FadeOut(self.tree_stump_group),
            FadeOut(continue_text),
            run_time=1
        )
        
        # Update title
        new_title = Text("Full Complexity Decision Tree", font_size=40, weight=BOLD, color=RED)
        new_title.to_edge(UP)
        self.play(Transform(self.title, new_title))
        
        # Fit full tree
        full_tree = DecisionTreeClassifier(criterion='gini', ccp_alpha=0, random_state=42)
        full_tree.fit(self.X, self.y)
        
        # Build tree with better spacing
        self.node_mobjects = {}
        self.grow_tree_with_better_spacing(full_tree, scale=0.7)
        
        self.wait(3)
    
    def grow_tree_with_better_spacing(self, tree, scale=0.7):
        """Tree growing with ADAPTIVE spacing for long single-child chains"""
        tree_structure = tree.tree_
        
        tree_depth = tree.get_depth()
        n_leaves = tree.get_n_leaves()
        
        initial_width = min(5.0, n_leaves * 0.4)
        
        def count_chain_depth(node_id):
            """Count how deep a single-child chain goes"""
            left_child = tree_structure.children_left[node_id]
            right_child = tree_structure.children_right[node_id]
            
            if left_child == -1 and right_child == -1:
                return 0
            
            # Check if only one child
            if left_child == -1 or right_child == -1:
                active_child = left_child if right_child == -1 else right_child
                return 1 + count_chain_depth(active_child)
            else:
                return 0
        
        def animate_node(node_id, x, y, width, depth=0):
            """Recursively animate node with adaptive vertical spacing"""
            is_leaf = tree_structure.feature[node_id] == -2
            
            if is_leaf:
                # Leaf node
                value = np.argmax(tree_structure.value[node_id])
                samples = tree_structure.n_node_samples[node_id]
                
                color = BLUE if value == 0 else RED
                
                shape = Circle(
                    radius=0.2 * scale,
                    color=color,
                    fill_opacity=0.7,
                    stroke_width=2,
                    stroke_color=WHITE
                )
                shape.move_to([x, y, 0])
                
                main_label = VGroup(
                    Text(f"C{value}", font_size=int(10*scale),
                         weight=BOLD, color=WHITE),
                    Text(f"n={samples}", font_size=int(8*scale), 
                         color=WHITE)
                ).arrange(DOWN, buff=0.02*scale)
                main_label.move_to(shape.get_center())
                
                self.node_mobjects[node_id] = {
                    'shape': shape,
                    'label': main_label,
                    'is_leaf': True,
                    'x': x,
                    'y': y
                }
                
                self.play(FadeIn(shape, scale=0.3), Write(main_label), 
                         run_time=0.1)
                
            else:
                # Decision node
                feature = tree_structure.feature[node_id]
                threshold = tree_structure.threshold[node_id]
                samples = tree_structure.n_node_samples[node_id]
                
                shape = RoundedRectangle(
                    width=1.2 * scale,
                    height=0.6 * scale,
                    corner_radius=0.05 * scale,
                    color=ORANGE,
                    fill_opacity=0.4,
                    stroke_width=2
                )
                shape.move_to([x, y, 0])
                
                main_label = VGroup(
                    Text(f"X{feature} ≤ {threshold:.2f}", font_size=int(9*scale), weight=BOLD),
                    Text(f"n={samples}", font_size=int(8*scale))
                ).arrange(DOWN, buff=0.02*scale)
                
                main_label.move_to(shape.get_center())
                
                self.node_mobjects[node_id] = {
                    'shape': shape,
                    'label': main_label,
                    'is_leaf': False,
                    'x': x,
                    'y': y,
                }
                
                self.play(Create(shape), Write(main_label), 
                         run_time=0.1)
                
                # Create children with ADAPTIVE spacing
                left_child = tree_structure.children_left[node_id]
                right_child = tree_structure.children_right[node_id]
                
                # Check if this is a long chain
                left_chain_depth = count_chain_depth(left_child) if left_child != -1 else 0
                right_chain_depth = count_chain_depth(right_child) if right_child != -1 else 0
                max_chain = max(left_chain_depth, right_chain_depth)
                
                # ADAPTIVE vertical spacing: much tighter for long chains
                if max_chain > 5:
                    vertical_spacing = 0.35 * scale  # Very tight for very long chains
                elif max_chain > 3:
                    vertical_spacing = 0.45 * scale  # Tight for long chains
                else:
                    vertical_spacing = 0.7 * scale  # Normal spacing
                
                new_width = width * 0.5
                child_y = y - vertical_spacing
                left_x = x - width * scale
                right_x = x + width * scale
                
                # Create edges
                if left_child != -1:
                    left_edge = Line(
                        [x, y - 0.25*scale, 0],
                        [left_x, child_y + 0.25*scale, 0],
                        color=BLUE,
                        stroke_width=2
                    )
                    
                    self.play(Create(left_edge), run_time=0.05)
                    animate_node(left_child, left_x, child_y, new_width, depth + 1)
                
                if right_child != -1:
                    right_edge = Line(
                        [x, y - 0.25*scale, 0],
                        [right_x, child_y + 0.25*scale, 0],
                        color=RED,
                        stroke_width=2
                    )
                    
                    self.play(Create(right_edge), run_time=0.05)
                    animate_node(right_child, right_x, child_y, new_width, depth + 1)
        
        # Start from root
        animate_node(0, 0, 2.2, initial_width)
        self.wait(0.5)

    def overfitting_alert(self):
        # we want to create an eliptic box (3.5 height, 2 width) in yellow
        # and we want to have a warning sign with text "Overfitting!"
        alert_box = Ellipse(
            width=1.75, height=5,
            color=YELLOW,
            stroke_width=3,
            fill_opacity=0.1
        )
        warning_sign = Text("⚠️ Overfitting!", font_size=36, color=YELLOW)
        alert_box.to_edge(DOWN).shift(LEFT * 1 + DOWN *1)
        warning_sign.next_to(alert_box, RIGHT, buff=0.5).shift(DOWN * 1)
        self.play(Write(alert_box), Write(warning_sign), run_time=1.5)
# End of file
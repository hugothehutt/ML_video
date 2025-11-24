from manim import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# ============================================================================
# SCENE 11: Boosting Concept - Sequential Learning (RESTRUCTURED)
# ============================================================================
class BoostingConceptScene(Scene):
    def construct(self):
        # Title
        title = Text("Boosting: Sequential Learning", font_size=44, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Initial subtitle
        subtitle = Text("Step-by-step AdaBoost visualization", 
                       font_size=24, color=YELLOW)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(subtitle))
        self.wait(0.5)
        
        # =================================================================
        # Generate L-shaped dataset (larger scale to fill square better)
        # =================================================================
        np.random.seed(42)
        n_points = 30
        
        # Create lower-L (red class, label 0) - fill more of the square
        lower_L_points = []
        # Horizontal part of lower L (bottom area, full width)
        for _ in range(11):
            x = np.random.uniform(-2.5, 2.5)
            y = np.random.uniform(-2.5, -0.2)
            lower_L_points.append([x, y])
        # Vertical part of lower L (left side, full height)
        for _ in range(6):
            x = np.random.uniform(-2.5, -0.2)
            y = np.random.uniform(-0.2, 2.5)
            lower_L_points.append([x, y])
        
        # Create upper-L (blue class, label 1) - fill more of the square
        upper_L_points = []
        # Horizontal part of upper L (top area, full width)
        for _ in range(11):
            x = np.random.uniform(-2.5, 2.5)
            y = np.random.uniform(0.2, 2.5)
            upper_L_points.append([x, y])
        # Vertical part of upper L (right side, full height)
        for _ in range(6):
            x = np.random.uniform(0.2, 2.5)
            y = np.random.uniform(-2.5, 0.2)
            upper_L_points.append([x, y])
        
        # Add some outliers (crossing into other class territory)
        lower_L_points.append([1.8, 1.8])  # red outlier in blue territory
        upper_L_points.append([-1.8, -1.8])  # blue outlier in red territory
        
        # Add minimal noise to keep classes separate
        lower_L_points = np.array(lower_L_points) + np.random.randn(len(lower_L_points), 2) * 0.08
        upper_L_points = np.array(upper_L_points) + np.random.randn(len(upper_L_points), 2) * 0.08
        
        # Combine into dataset
        X_train = np.vstack([lower_L_points, upper_L_points])
        y_train = np.array([0] * len(lower_L_points) + [1] * len(upper_L_points))
        
        # Initialize weights uniformly (will be normalized after first error calculation)
        n_samples = len(X_train)
        sample_weights = np.ones(n_samples) / n_samples
        
        # Store for later use
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weights = sample_weights.copy()
        
        # =================================================================
        # Create formula box in bottom center (below data visualization)
        # =================================================================
        formula_box = RoundedRectangle(
            width=4, height=2.2, corner_radius=0.15,
            stroke_width=3, color=TEAL, fill_opacity=0.08
        )
        formula_box.move_to(DOWN * 2.8)  # Bottom center position
        
        formula_title = Text("AdaBoost Formulas", font_size=18, weight=BOLD, color=TEAL)
        formula_title.next_to(formula_box.get_top(), DOWN, buff=0.15)
        
        # shift Error and Weight to the left of the box
        error_formula = VGroup(
            Text("Error:", font_size=14, color=WHITE),
            MathTex(r"\epsilon = ", font_size=26),
            MathTex(r"\sum_{i: h(x_i) \neq y_i} w_i", font_size=26)
        ).arrange(RIGHT, buff=0.1)
        error_formula.next_to(formula_title, DOWN, buff=0.2).shift(LEFT * 0.8)
        
        alpha_formula = VGroup(
            Text("Weight:", font_size=14, color=WHITE),
            MathTex(r"\alpha = \frac{1}{2} \ln\left(\frac{1-\epsilon}{\epsilon}\right)", font_size=26)
        ).arrange(RIGHT, buff=0.1)
        # align on left edge of Error formula
        alpha_formula.next_to(error_formula, DOWN, buff=0.2).align_to(error_formula, LEFT)
        
        self.play(Create(formula_box), run_time=0.5)
        self.play(Write(formula_title), Write(error_formula), Write(alpha_formula), run_time=1)
        self.wait(0.5)
        
        # Store formula box for later
        self.formula_box_group = VGroup(formula_box, formula_title, error_formula, alpha_formula)
        
        # =================================================================
        # Show initial data in center (fixed position)
        # =================================================================
        self.update_subtitle(subtitle, "Round 1: Initial dataset")
        
        # Center data visualization (will stay fixed)
        center_data_box = self.create_data_visualization(
            X_train, y_train, sample_weights, 
            center=ORIGIN,
            show_weights=False
        )

        self.play(FadeIn(center_data_box, scale=0.95), run_time=1)
        self.wait(0.5)
        
        # Store the data box and dots for later weight updates
        self.center_square = center_data_box[0]
        self.center_dots = center_data_box[1]
        
        # =================================================================
        # BOOSTING ROUNDS (3 rounds)
        # =================================================================
        n_rounds = 3
        alpha_box_combos = []  # Store for final sum
        round_labels = []  # Store round labels
        
        for round_idx in range(n_rounds):
            round_num = round_idx + 1
            
            # Update subtitle for current step
            self.update_subtitle(subtitle, f"Round {round_num}: Fitting decision stump")
            
            # Calculate vertical position for this round (smaller buff between rounds)
            round_y_offset = 2 - round_idx * 1.8  # Start higher (+1), smaller spacing
            
            # LANE 1 (Left): Round number
            round_label = Text(f"Round {round_num}", font_size=28, weight=BOLD, color=ORANGE)
            round_label.move_to(LEFT * 5.5 + UP * round_y_offset)
            self.play(Write(round_label), run_time=0.5)
            round_labels.append(round_label)
            
            # ============================================================
            # STEP 1: Fit decision stump and show decision boundary
            # ============================================================
            self.update_subtitle(subtitle, f"Round {round_num}: Step 1 - Fit decision stump")
            
            # Train a depth-1 tree (decision stump) with sample weights
            stump = DecisionTreeClassifier(max_depth=1, random_state=42 + round_idx)
            stump.fit(X_train, y_train, sample_weight=self.sample_weights)
            
            # Get decision boundary info
            feature = stump.tree_.feature[0]
            threshold = stump.tree_.threshold[0]
            
            # Create decision boundary visualization (in center, fixed)
            boundary_viz = self.create_decision_boundary(feature, threshold, center=ORIGIN)
            self.play(FadeIn(boundary_viz, scale=0.98), run_time=1)
            self.wait(0.8)
            
            # ============================================================
            # STEP 2: Calculate ERROR first, then alpha
            # ============================================================
            self.update_subtitle(subtitle, f"Round {round_num}: Step 2 - Calculate error and α")
            
            # Make predictions
            y_pred = stump.predict(X_train)
            
            # Calculate weighted error FIRST
            incorrect = (y_pred != y_train)
            weighted_error = np.sum(self.sample_weights[incorrect])
            
            # Display error value in right lane
            error_value = Text(f"ε = {weighted_error:.3f}", font_size=20, color=RED, weight=BOLD)
            error_value.move_to(RIGHT * 5 + UP * (round_y_offset + 0.4))
            self.play(Write(error_value), run_time=0.6)
            self.wait(0.5)
            
            # Calculate alpha (AdaBoost formula)
            epsilon = 1e-10
            weighted_error = np.clip(weighted_error, epsilon, 1 - epsilon)
            alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
            
            # Display alpha value
            alpha_value = Text(f"α{round_num} = {alpha:.3f}", font_size=22, color=GREEN, weight=BOLD)
            alpha_value.move_to(RIGHT * 5 + UP * (round_y_offset - 0.2))
            self.play(Write(alpha_value), run_time=0.6)
            self.wait(0.8)
            
            # Fade away both values
            self.play(FadeOut(error_value), FadeOut(alpha_value), run_time=0.5)
            
            # Create alpha × [colored box] combo
            alpha_symbol = MathTex(f"\\alpha_{round_num}", font_size=24, color=GREEN)
            
            # Create a copy of the colored boundary box (smaller)
            boundary_copy = boundary_viz.copy()
            boundary_copy.scale(0.35)
            
            times_symbol = MathTex(r"\times", font_size=20)
            
            # Arrange: α × [box]
            combo_group = VGroup(alpha_symbol, times_symbol, boundary_copy)
            combo_group.arrange(RIGHT, buff=0.12)
            combo_group.move_to(RIGHT * 5 + UP * round_y_offset)
            
            # Animate: move colored box and alpha symbol simultaneously
            self.play(
                TransformFromCopy(boundary_viz, boundary_copy),
                Write(alpha_symbol),
                Write(times_symbol),
                run_time=1
            )
            self.wait(0.5)
            
            # Store this combo for final sum (keep it on screen!)
            alpha_box_combos.append(combo_group)
            
            # ============================================================
            # STEP 3: Update weights and visualize in the data
            # ============================================================
            if round_idx < n_rounds - 1:  # Don't update after last round
                self.update_subtitle(subtitle, f"Round {round_num}: Step 3 - Update sample weights")
                
                # Update sample weights (AdaBoost formula)
                # Increase weight for misclassified, decrease for correct
                # w_i := w_i * exp(alpha) if incorrect, w_i * exp(-alpha) if correct
                weight_multiplier = np.where(incorrect, np.exp(alpha), np.exp(-alpha))
                self.sample_weights *= weight_multiplier
                # Normalize
                self.sample_weights /= np.sum(self.sample_weights)
                
                # Highlight misclassified points
                highlights = self.visualize_weight_update(X_train, y_train, incorrect, center=ORIGIN)
                self.play(FadeIn(highlights), run_time=0.6)
                self.wait(0.5)
                
                # Update dot sizes to reflect new weights
                new_dots = VGroup()
                for i, (point, label) in enumerate(zip(X_train, y_train)):
                    x_scaled = point[0] * 0.45  # Consistent scaling
                    y_scaled = point[1] * 0.45
                    pos = ORIGIN + RIGHT * x_scaled + UP * y_scaled
                    
                    color = RED if label == 0 else BLUE
                    base_radius = 0.06
                    weight_scale = self.sample_weights[i] * len(self.sample_weights)
                    radius = base_radius * (0.5 + weight_scale)
                    
                    dot = Dot(pos, color=color, radius=radius)
                    new_dots.add(dot)
                new_dots.shift(UP*0.3)
                
                # Animate size changes
                self.play(
                    Transform(self.center_dots, new_dots),
                    FadeOut(highlights),
                    run_time=1
                )
                self.wait(0.5)
                
                # Fade out decision boundary (but keep data visible)
                self.play(FadeOut(boundary_viz), run_time=0.5)
            else:
                # Last round: just fade out boundary
                self.play(FadeOut(boundary_viz), run_time=0.5)
        
        # =================================================================
        # FINAL: Combine all terms into ensemble
        # =================================================================
        self.update_subtitle(subtitle, "Final Ensemble: Combining all weak learners")
        self.wait(0.5)
        
        # Fade out everything EXCEPT: title, subtitle, and alpha combos
        objects_to_keep = [title, subtitle] + alpha_box_combos
        objects_to_fade = []
        for mob in self.mobjects:
            is_kept = False
            for kept in objects_to_keep:
                if mob == kept or mob in kept.submobjects:
                    is_kept = True
                    break
            if not is_kept:
                objects_to_fade.append(mob)
        
        self.play(*[FadeOut(mob) for mob in objects_to_fade], run_time=1)
        self.wait(0.3)
        
        # Move alpha combos DIRECTLY to their final horizontal positions (no vertical stacking)
        # Calculate final positions
        final_x_positions = [-3.5, 0, 3.5]  # Spread out horizontally
        final_y = 1.2
        
        move_anims = []
        for i, combo in enumerate(alpha_box_combos):
            target_pos = np.array([final_x_positions[i], final_y, 0])
            move_anims.append(combo.animate.move_to(target_pos))
        
        self.play(*move_anims, run_time=1.2)
        self.wait(0.4)
        
        # Add plus signs between the combos
        plus_signs = []
        for i in range(len(alpha_box_combos) - 1):
            plus = MathTex("+", font_size=28, color=WHITE)
            # Position between combos
            x_pos = (final_x_positions[i] + final_x_positions[i + 1]) / 2
            plus.move_to(np.array([x_pos, final_y, 0]))
            plus_signs.append(plus)
        
        self.play(*[Write(plus) for plus in plus_signs], run_time=0.5)
        self.wait(0.5)
        
        # Add equals sign ROTATED 90 degrees (vertical) below the sum
        equals = MathTex("=", font_size=32)
        equals.rotate(90 * DEGREES)
        equals.move_to(DOWN * 0.1)
        
        self.play(Write(equals), run_time=0.5)
        self.wait(0.3)
        
        # Create final ensemble boundary below the equals
        final_boundary = self.create_final_ensemble_boundary()
        final_boundary.scale(0.45)
        final_boundary.move_to(DOWN * 1.5)
        
        self.play(FadeIn(final_boundary, scale=0.95), run_time=1)
        self.wait(0.8)
        
        # Bottom text
        bottom_text = Text("We built a Boosting Classifier!", 
                          font_size=32, color=GREEN, weight=BOLD)
        bottom_text.move_to(DOWN * 2.8)
        self.play(Write(bottom_text), run_time=1)
        
        self.wait(3)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def update_subtitle(self, subtitle_mob, new_text):
        """Update the subtitle text"""
        new_subtitle = Text(new_text, font_size=24, color=YELLOW)
        new_subtitle.move_to(subtitle_mob.get_center())
        self.play(Transform(subtitle_mob, new_subtitle), run_time=0.4)
        self.wait(0.3)
    
    def create_data_visualization(self, X, y, weights, center=ORIGIN, show_weights=False):
        """
        Create a visualization of the dataset with optional weight visualization
        Returns a VGroup containing the square boundary and data points
        """
        # Create square boundary (reduced size)
        square = Square(side_length=3.5, color=WHITE, stroke_width=2)
        square.move_to(center)
        
        # Create data points scaled to fit in the square
        dots = VGroup()
        for i, (point, label) in enumerate(zip(X, y)):
            # Scale coordinates from data space to square space
            x_scaled = point[0] * 0.45  # Back to original scaling
            y_scaled = point[1] * 0.45
            pos = center + RIGHT * x_scaled + UP * y_scaled
            
            # Color by class
            color = RED if label == 0 else BLUE
            
            # Size by weight if showing weights
            if show_weights:
                # Scale radius based on weight (relative to uniform weight)
                base_radius = 0.06
                weight_scale = weights[i] * len(weights)  # Relative to uniform
                radius = base_radius * (0.5 + weight_scale)
            else:
                radius = 0.06
            
            dot = Dot(pos, color=color, radius=radius)
            dots.add(dot)
        
        return VGroup(square, dots).shift(UP*0.3)
    
    def create_decision_boundary(self, feature, threshold, center=ORIGIN):
        """
        Create colored rectangles showing the decision boundary
        feature: 0 (vertical split) or 1 (horizontal split)
        threshold: split value
        Returns a VGroup with two colored rectangles
        """
        # Scale threshold to square coordinates
        threshold_scaled = threshold * 0.45
        square_half = 1.75  # Half of 3.5
        
        if feature == 0:  # Vertical split (split on X)
            # Left rectangle (predicted class depends on training data, assume RED=0)
            left_width = square_half + threshold_scaled
            left_rect = Rectangle(width=left_width, height=3.5,
                                 color=RED, fill_opacity=0.3, stroke_width=0)
            left_rect.move_to(center + LEFT * (square_half - left_width/2))
            
            # Right rectangle (assume BLUE=1)
            right_width = square_half - threshold_scaled
            right_rect = Rectangle(width=right_width, height=3.5,
                                  color=BLUE, fill_opacity=0.3, stroke_width=0)
            right_rect.move_to(center + RIGHT * (square_half - right_width/2))
            
            # Decision line
            decision_line = Line(
                center + UP * square_half + RIGHT * threshold_scaled,
                center + DOWN * square_half + RIGHT * threshold_scaled,
                color=YELLOW, stroke_width=4
            )
            
            return VGroup(left_rect, right_rect, decision_line).shift(UP*0.3)
        
        else:  # Horizontal split (split on Y)
            # Bottom rectangle (assume RED=0)
            bottom_height = square_half + threshold_scaled
            bottom_rect = Rectangle(width=3.5, height=bottom_height,
                                   color=RED, fill_opacity=0.3, stroke_width=0)
            bottom_rect.move_to(center + DOWN * (square_half - bottom_height/2))
            
            # Top rectangle (assume BLUE=1)
            top_height = square_half - threshold_scaled
            top_rect = Rectangle(width=3.5, height=top_height,
                                color=BLUE, fill_opacity=0.3, stroke_width=0)
            top_rect.move_to(center + UP * (square_half - top_height/2))
            
            # Decision line
            decision_line = Line(
                center + LEFT * square_half + UP * threshold_scaled,
                center + RIGHT * square_half + UP * threshold_scaled,
                color=YELLOW, stroke_width=4
            )
            
            return VGroup(bottom_rect, top_rect, decision_line).shift(UP*0.3)
    
    def visualize_weight_update(self, X, y, incorrect, center=ORIGIN):
        """
        Highlight misclassified points that will get higher weights
        """
        highlights = VGroup()
        for i, is_wrong in enumerate(incorrect):
            if is_wrong:
                x_scaled = X[i, 0] * 0.45
                y_scaled = X[i, 1] * 0.45
                pos = center + RIGHT * x_scaled + UP * y_scaled
                
                # Create pulsing circle around misclassified point
                circle = Circle(radius=0.15, color=YELLOW, stroke_width=3)
                circle.move_to(pos)
                highlights.add(circle)
        
        return highlights.shift(UP*0.3)
    
    def create_final_ensemble_boundary(self):
        """
        Create a visual representation of the final ensemble decision boundary
        This is a simplified representation showing a more complex boundary
        """
        square = Square(side_length=4, color=WHITE, stroke_width=2)
        
        # Create a more complex decision boundary (simplified visualization)
        # In reality, you'd need to evaluate the ensemble on a grid
        # For now, show a conceptual "curved" boundary using polygons
        
        # Lower-L region (RED)
        lower_L = Polygon(
            [-2, -2, 0], [2, -2, 0], [2, -0.5, 0], 
            [0, -0.5, 0], [0, 2, 0], [-2, 2, 0],
            color=RED, fill_opacity=0.3, stroke_width=0
        )
        
        # Upper-L region (BLUE)
        upper_L = Polygon(
            [0, -0.5, 0], [2, -0.5, 0], [2, 2, 0], 
            [-2, 2, 0], [-2, 0.5, 0], [0, 0.5, 0],
            color=BLUE, fill_opacity=0.3, stroke_width=0
        )
        
        return VGroup(square, lower_L, upper_L)


# ============================================================================
# SCENE: Enhanced Boosting Decision Boundary Evolution
# ============================================================================
class BoostingBoundary(Scene):
    def construct(self):
        # Title
        title = Text("AdaBoost: Decision Boundary Evolution", 
                    font_size=44, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Generate data (matching your project standard)
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        self.X = X
        self.y = y
        self.X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 8 + 1
        
        # =================================================================
        # Create coordinate system (right side, matching your style)
        # =================================================================
        self.axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=5.5,
            y_length=5.5,
            axis_config={"include_tip": False, "font_size": 18}
        )
        # Shift right, and we'll align with stats box later
        self.axes.shift(RIGHT * 2.5 + DOWN * 0.3)
        
        # Create axis labels and position them OUTSIDE (below x-axis, left of y-axis)
        x_label = Text("Feature 1", font_size=18)
        x_label.next_to(self.axes.x_axis, DOWN, buff=0.3)
        
        y_label = Text("Feature 2", font_size=18)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(self.axes.y_axis, LEFT, buff=0.3)
        
        axes_labels = VGroup(x_label, y_label)
        
        self.play(Create(self.axes), Write(axes_labels), run_time=0.9)
        self.axes_labels = axes_labels
        self.wait(0.3)
        
        # =================================================================
        # Show data points (using your PURE_BLUE and PURE_RED scheme)
        # =================================================================
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
            self.data_dots.add(dot)
        
        self.data_dots.set_z_index(10)
        self.play(
            LaggedStart(*[GrowFromCenter(dot) for dot in self.data_dots], 
                       lag_ratio=0.015),
            run_time=1.5
        )
        self.wait(0.5)
        
        # =================================================================
        # Create stats box (lower left, MOVED UP 0.1)
        # =================================================================
        self.stats_box = Rectangle(width=3.5, height=2.5, color=WHITE, stroke_width=2, fill_opacity=0.05)
        self.stats_box.to_corner(DL).shift(UP * 0.4 + RIGHT * 0.3)  # Changed from 0.3 to 0.4
        
        stats_title = Text("Model Statistics", font_size=20, weight=BOLD, color=GREEN)
        stats_title.next_to(self.stats_box.get_top(), DOWN, buff=0.15)
        
        self.stats_content = VGroup(
            Text("Estimators: --", font_size=18),
            Text("Accuracy: --", font_size=18),
            Text("Train Score: --", font_size=18),
            Text("Weak learners: depth=1", font_size=16, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        self.stats_content.next_to(stats_title, DOWN, buff=0.2)
        
        # NOW align axes x-axis with bottom of stats box
        stats_bottom_y = self.stats_box.get_bottom()[1]
        axes_bottom_y = self.axes.get_bottom()[1]
        shift_amount = axes_bottom_y - stats_bottom_y
        self.stats_box.shift(UP*shift_amount)
        self.stats_content.shift(UP*shift_amount)
        stats_title.shift(UP*shift_amount)

        self.play(Create(self.stats_box), Write(stats_title), run_time=0.6)
        self.play(Write(self.stats_content), run_time=0.5)
        self.stats_title = stats_title
        self.wait(0.3)

        # =================================================================
        # Create status text box (MOVED DOWN 0.5 from previous)
        # =================================================================
        self.status_text = Text("Building AdaBoost model...", font_size=22, color=YELLOW)
        self.status_text.next_to(self.stats_box, UP, buff=1.5)  # Changed from 2.0 to 1.5
        self.status_text.align_to(self.stats_box, LEFT)
        self.play(Write(self.status_text), run_time=0.5)
        self.wait(0.3)
        
        # =================================================================
        # Create confidence legend (RIGHT side of axes, MOVED DOWN 0.2)
        # =================================================================
        legend = self.create_confidence_legend()
        # Position to the right of the axes, aligned with axes
        legend.next_to(self.axes, RIGHT, buff=0.5)
        legend.align_to(self.axes, DOWN)
        legend.shift(DOWN * 0.2)  # Move down 0.2
        self.play(FadeIn(legend, scale=0.95), run_time=0.7)
        self.legend = legend
        self.wait(0.3)
        
        # =================================================================
        # Initialize boundary blocks storage
        # =================================================================
        self.boundary_blocks = None
        
        # =================================================================
        # Progressive AdaBoost evolution
        # =================================================================
        estimators_list = [1, 2, 3, 5, 10, 20, 50, 150]
        
        for idx, n_estimators in enumerate(estimators_list):
            # Update status text
            status_msg = f"AdaBoost with {n_estimators} weak learner{'s' if n_estimators > 1 else ''}"
            self.update_status_text(status_msg)
            
            # Train AdaBoost model
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n_estimators,
                random_state=42,
                algorithm='SAMME'
            )
            model.fit(X, y)
            
            # Update boundary with morphing animation
            self.update_boundary_with_morphing(model, n_estimators)
            
            # Update statistics
            accuracy = model.score(X, y)
            self.update_stats(n_estimators, accuracy)
            
            self.wait(1.5 if idx < len(estimators_list) - 1 else 2)
        
        self.wait(3)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def update_status_text(self, new_text):
        """Update status text above stats box"""
        new_status = Text(new_text, font_size=22, color=YELLOW)
        new_status.move_to(self.status_text.get_center())
        new_status.align_to(self.stats_box, LEFT)
        self.play(Transform(self.status_text, new_status), run_time=0.4)
        self.wait(0.2)
    
    def update_stats(self, n_estimators, accuracy):
        """Update statistics display"""
        new_stats = VGroup(
            Text(f"Estimators: {n_estimators}", font_size=18, color=GREEN if n_estimators >= 10 else WHITE),
            Text(f"Accuracy: {accuracy:.1%}", font_size=18, color=GREEN if accuracy > 0.95 else YELLOW),
            Text(f"Train Score: {accuracy:.3f}", font_size=18),
            Text(f"Weak learners: depth=1", font_size=16, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        new_stats.move_to(self.stats_content.get_center())
        self.play(Transform(self.stats_content, new_stats), run_time=0.5)
    
    def update_boundary_with_morphing(self, model, n_estimators):
        """Update decision boundary with morphing and white flash on changes"""
        # Fine grid for smooth boundaries
        h = 0.05
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.3, self.X[:, 1].max() + 0.3
        
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )
        
        # Get predictions and probabilities
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Get decision function for confidence
        try:
            Z_confidence = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z_confidence = Z_confidence.reshape(xx.shape)
            # Normalize confidence to [0, 1]
            Z_prob = 1 / (1 + np.exp(-Z_confidence))
        except:
            # Fallback if decision_function not available
            Z_prob = np.where(Z == 0, 0.2, 0.8)
        
        # Normalize coordinates
        xx_norm = (xx - self.X[:, 0].min()) / (self.X[:, 0].max() - self.X[:, 0].min()) * 8 + 1
        yy_norm = (yy - self.X[:, 1].min()) / (self.X[:, 1].max() - self.X[:, 1].min()) * 8 + 1
        
        # First time: create boundary blocks
        if self.boundary_blocks is None:
            self.boundary_blocks = {}
            
            for i in range(0, len(xx) - 1, 1):  # Skip for performance
                for j in range(0, len(yy[0]) - 1, 1):
                    color = self.get_color_from_prob(Z_prob[i, j])
                    opacity = self.get_opacity_from_prob(Z_prob[i, j])
                    
                    square = Square(
                        side_length=h * 2 * 5.5/10,
                        color=color,
                        fill_opacity=opacity,
                        stroke_width=0
                    )
                    square.move_to(self.axes.c2p(xx_norm[i, j], yy_norm[i, j]))
                    square.set_z_index(1)
                    
                    self.boundary_blocks[(i, j)] = {
                        'square': square,
                        'current_prob': Z_prob[i, j],
                        'current_class': Z[i, j]
                    }
            
            # Animate initial boundary
            all_squares = VGroup(*[block['square'] for block in self.boundary_blocks.values()])
            self.play(FadeIn(all_squares, scale=1.02), run_time=1.2)
            self.bring_to_front(self.data_dots)
        
        else:
            # Subsequent times: morph changed blocks
            changed_info = []
            
            for i in range(0, len(xx) - 1, 1):
                for j in range(0, len(yy[0]) - 1, 1):
                    if (i, j) in self.boundary_blocks:
                        block_info = self.boundary_blocks[(i, j)]
                        square = block_info['square']
                        old_prob = block_info['current_prob']
                        old_class = block_info['current_class']
                        new_prob = Z_prob[i, j]
                        new_class = Z[i, j]
                        
                        # Check for significant change (class change or >10% probability change)
                        if old_class != new_class or abs(new_prob - old_prob) > 0.1:
                            new_color = self.get_color_from_prob(new_prob)
                            new_opacity = self.get_opacity_from_prob(new_prob)
                            changed_info.append({
                                'square': square,
                                'new_color': new_color,
                                'new_opacity': new_opacity,
                                'new_prob': new_prob,
                                'new_class': new_class,
                                'key': (i, j)
                            })
            
            # Animate changes with white flash
            if changed_info:
                # Step 1: White flash
                flash_anims = [
                    info['square'].animate.set_fill(WHITE, opacity=0.7)
                    for info in changed_info
                ]
                self.play(*flash_anims, run_time=0.3)
                
                # Step 2: Transition to new colors
                color_anims = [
                    info['square'].animate.set_fill(info['new_color'], opacity=info['new_opacity'])
                    for info in changed_info
                ]
                self.play(*color_anims, run_time=0.9)
                
                # Step 3: Update stored values
                for info in changed_info:
                    self.boundary_blocks[info['key']]['current_prob'] = info['new_prob']
                    self.boundary_blocks[info['key']]['current_class'] = info['new_class']
                
                self.bring_to_front(self.data_dots)
            else:
                self.wait(0.5)
    
    def get_color_from_prob(self, prob):
        """Get color based on probability (using BRIGHTER base colors)"""
        if prob < 0.5:
            # Brighter blue - interpolate from light blue to pure blue
            return interpolate_color(BLUE_C, BLUE_B, prob * 2)
        else:
            # Brighter red - interpolate from pure red to darker red
            return interpolate_color(RED_B, RED_C, (prob - 0.5) * 2)
    
    def get_opacity_from_prob(self, prob):
        """Get opacity based on confidence (higher range for better visibility)"""
        confidence = abs(prob - 0.5) * 2  # 0 at boundary, 1 at extremes
        return 0.35 + confidence * 0.45  # Range: 0.35 to 0.8 (much brighter!)
    
    def create_confidence_legend(self):
        """Create VERTICAL confidence gradient legend - NO box, NO title, labels outside"""
        # Create vertical gradient bars (many bars for smooth gradient)
        n_bars = 30  # More bars for smoother gradient
        bar_height = 5.5 / n_bars  # Match axes height (5.5)
        
        gradient_bars = VGroup()
        for i in range(n_bars):
            # Probability from 0 (bottom) to 1 (top)
            prob = i / (n_bars - 1)
            color = self.get_color_from_prob(prob)
            opacity = self.get_opacity_from_prob(prob)
            
            bar = Rectangle(
                width=0.4,
                height=bar_height,
                color=color,
                fill_opacity=opacity,
                stroke_width=0.5,
                stroke_color=WHITE
            )
            # Stack vertically from bottom to top
            bar.shift(UP * i * bar_height)
            gradient_bars.add(bar)
        
        # Center the gradient bars at origin
        gradient_bars.move_to(ORIGIN)
        
        # Labels OUTSIDE the gradient (above and below)
        label_class1 = Text("Class 1", font_size=14, weight=BOLD, color=RED)
        label_class1.next_to(gradient_bars, UP, buff=0.15)
        
        label_class0 = Text("Class 0", font_size=14, weight=BOLD, color=BLUE)
        label_class0.next_to(gradient_bars, DOWN, buff=0.15)
        
        # Return just the gradient and labels (no box, no title)
        return VGroup(gradient_bars, label_class1, label_class0)

# ============================================================================
# SCENE: Boosting vs Bagging Comparison (Updated with consistent colors)
# ============================================================================
class BoostingVsBaggingScene(Scene):
    def construct(self):
        # Title
        title = Text("Boosting vs Bagging: Side-by-Side", 
                    font_size=40, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Generate data
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        y_adaboost = 2 * y - 1
        
        # Store for later use
        self.X = X
        self.y = y
        
        # Models with consistent parameters
        models = [
            ("Random Forest\n(Bagging)", 
             RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42), 
             GREEN, y, "bagging"),
            ("AdaBoost\n(Boosting)", 
             AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), 
                               n_estimators=50, random_state=42, algorithm='SAMME'), 
             ORANGE, y_adaboost, "boosting")
        ]
        
        # Train models
        for name, model, color, y_train, model_type in models:
            model.fit(X, y_train)
        
        # Create two subplots
        positions = [LEFT * 4.1, RIGHT * 4.1]  # Changed from 3.5 to 4.1 (0.6 wider)
        
        all_visuals = []
        
        for (name, model, color, y_train, model_type), x_pos in zip(models, positions):
            # Use original normalization to [1, 5] range (4 + 1)
            X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 4 + 1
            
            # Title (moved down 1.3)
            model_title = Text(name, font_size=22, weight=BOLD, color=color)
            model_title.move_to(x_pos + UP * 2.5)  # Changed from UP * 3 to UP * 1.7
            
            # Axes - ORIGINAL SIZE: 4x4 with range [0, 6]
            axes = Axes(
                x_range=[0, 6, 2],
                y_range=[0, 6, 2],
                x_length=4,
                y_length=4,
                axis_config={"include_tip": False, "font_size": 16}
            )
            axes.move_to(x_pos + UP*0.2)  # Changed from DOWN * 0.3 to DOWN * 1.3
            
            # Boundary with consistent colors
            boundary = self.create_boundary(axes, model, X, X_normalized, model_type)
            
            # Data points with PURE_BLUE and PURE_RED (NEW STYLE)
            dots = VGroup()
            for point, label in zip(X_normalized, y):
                dot = Dot(
                    axes.c2p(point[0], point[1]),
                    color=PURE_BLUE if label == 0 else PURE_RED,
                    radius=0.06,
                    stroke_width=1.5,
                    stroke_color=WHITE,
                    fill_opacity=1.0
                )
                dots.add(dot)
            dots.set_z_index(10)
            
            # Stats (buff 0.5 below axes)
            if model_type == "bagging":
                accuracy = model.score(X, y)
            else:
                accuracy = model.score(X, y_adaboost)
            
            stats = VGroup(
                Text(f"Accuracy: {accuracy:.1%}", font_size=16, color=GREEN if accuracy > 0.95 else YELLOW),
                Text(f"Estimators: {model.n_estimators}", font_size=14),
                Text(f"Max depth: {'5' if model_type == 'bagging' else '1'}", font_size=14, color=GRAY)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            stats.next_to(axes, DOWN, buff=0.5)  # Changed to use buff=0.5
            
            all_visuals.append((model_title, axes, boundary, dots, stats))
        
        # Animate in sequence
        self.play(*[Write(v[0]) for v in all_visuals], run_time=1)
        self.play(*[Create(v[1]) for v in all_visuals], run_time=1)
        self.play(*[FadeIn(v[2], scale=1.02) for v in all_visuals], run_time=1.5)
        self.play(*[LaggedStart(*[GrowFromCenter(d) for d in v[3]], lag_ratio=0.01) 
                   for v in all_visuals], run_time=1.8)
        self.play(*[Write(v[4]) for v in all_visuals], run_time=1)
        
        self.wait(1)
        
        # Comparison table with better formatting
        comparison = VGroup(
            Text("Key Differences:", font_size=22, weight=BOLD, color=YELLOW),
            VGroup(
                Text("Bagging (Random Forest):", font_size=18, color=GREEN, weight=BOLD),
                Text("• Parallel training", font_size=15),
                Text("• Equal weights for all trees", font_size=15),
                Text("• Reduces variance (overfitting)", font_size=15),
                Text("• Random feature subsets", font_size=15)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15),
            VGroup(
                Text("Boosting (AdaBoost):", font_size=18, color=ORANGE, weight=BOLD),
                Text("• Sequential training", font_size=15),
                Text("• Adaptive weights on samples", font_size=15),
                Text("• Reduces bias (underfitting)", font_size=15),
                Text("• Focuses on hard examples", font_size=15)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        
        # Center horizontally and position at system level (around DOWN * 1.3)
        comparison.move_to(ORIGIN + UP*0.2)  # Centered at axes level
        box = SurroundingRectangle(comparison, color=WHITE, buff=0.25, corner_radius=0.12, stroke_width=2)
        
        self.play(Create(box), run_time=0.6)
        self.play(Write(comparison[0]), run_time=0.5)
        self.wait(0.3)
        
        # Animate bullet points
        for item in comparison[1:]:
            self.play(FadeIn(item, shift=UP*0.3), run_time=0.7)
            self.wait(0.4)
        
        self.wait(3)
    
    def create_boundary(self, axes, model, X_original, X_normalized, model_type):
        """Create boundary with ORIGINAL sizing from your code + consistent colors"""
        h = 0.04  # ORIGINAL value
        x_min, x_max = X_original[:, 0].min() - 0.5, X_original[:, 0].max() + 0.5
        y_min, y_max = X_original[:, 1].min() - 0.3, X_original[:, 1].max() + 0.3
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Get probabilities based on model type
        if model_type == "boosting":
            Z_decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z_prob = 1 / (1 + np.exp(-Z_decision.reshape(xx.shape)))
        else:  # bagging
            Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z_prob = Z_proba[:, 1].reshape(xx.shape)
        
        # ORIGINAL normalization to [1, 5] range
        xx_norm = (xx - X_original[:, 0].min()) / (X_original[:, 0].max() - X_original[:, 0].min()) * 4 + 1
        yy_norm = (yy - X_original[:, 1].min()) / (X_original[:, 1].max() - X_original[:, 1].min()) * 4 + 1
        
        regions = VGroup()
        for i in range(0, len(xx) - 1, 1):  # ORIGINAL: skip by 1 (no skipping)
            for j in range(0, len(yy[0]) - 1, 1):
                prob = Z_prob[i, j]
                
                # Use consistent color scheme
                color = self.get_color_from_prob(prob)
                opacity = self.get_opacity_from_prob(prob)
                
                square = Square(
                    side_length=h * 4/10 * 1.2,  # ORIGINAL sizing formula
                    color=color,
                    fill_opacity=opacity,
                    stroke_width=0
                )
                square.move_to(axes.c2p(xx_norm[i, j], yy_norm[i, j]))
                square.set_z_index(1)
                regions.add(square)
        
        return regions
    
    def get_color_from_prob(self, prob):
        """Get color based on probability (matching Enhanced scene)"""
        if prob < 0.5:
            # Brighter blue - interpolate from light blue to pure blue
            return interpolate_color(BLUE_C, BLUE_B, prob * 2)
        else:
            # Brighter red - interpolate from pure red to darker red
            return interpolate_color(RED_B, RED_C, (prob - 0.5) * 2)
    
    def get_opacity_from_prob(self, prob):
        """Get opacity based on confidence (matching Enhanced scene)"""
        confidence = abs(prob - 0.5) * 2  # 0 at boundary, 1 at extremes
        return 0.35 + confidence * 0.45  # Range: 0.35 to 0.8

# End of file
from manim import *
import numpy as np
import random
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# ==============================================================
# Bagging -> Random Forest FULL Scene (WITH CARD ANIMATION)
# ==============================================================
class BaggingScene(Scene):
    def construct(self):
        # TITLE
        title = Text("Bagging: Bootstrap Aggregating", font_size=44, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # STEP 1 — ORIGINAL DATASET AS CARDS
        step_text = Text("1) Dataset", font_size=26, color=YELLOW)
        step_text.next_to(title, DOWN, buff=0.3)
        self.play(Write(step_text))

        X, y = make_moons(n_samples=30, noise=0.3, random_state=42)
        original_data = VGroup()

        for i, (point, label) in enumerate(zip(X, y)):
            color = PURE_BLUE if label == 0 else PURE_RED
            
            # Create card
            card = RoundedRectangle(
                width=0.4,
                height=0.6,
                corner_radius=0.05,
                fill_opacity=1,
                fill_color=BLUE_E,
                stroke_color=WHITE,
                stroke_width=2
            )
            
            # Add index number (positioned lower)
            num_label = Text(str(i+1), font_size=20, color=WHITE)
            num_label.move_to(card.get_center() + DOWN * 0.05)
            
            # Add class indicator (color-coded dot at top)
            dot = Dot(radius=0.05, color=color)
            dot.move_to(card.get_top() + DOWN * 0.15)
            
            card_group = VGroup(card, num_label, dot)
            card_group.card_id = i + 1
            card_group.dot_color = color
            
            # Position in grid
            row = i // 6
            col = i % 6
            pos = LEFT * 1.5 + UP * (2 - row * 0.7) + RIGHT * (col * 0.6) + DOWN * 0.2
            card_group.move_to(pos)
            
            original_data.add(card_group)

        self.play(LaggedStart(*[FadeIn(g, scale=0.5) for g in original_data], lag_ratio=0.05), run_time=2)
        self.wait(1)

        # STEP 2 — BOOTSTRAP SAMPLES WITH THEATER ANIMATION
        new_step = Text("2) Bootstrap sampling (with replacement)", font_size=26, color=YELLOW)
        new_step.move_to(step_text.get_center())
        self.play(Transform(step_text, new_step))
        self.wait(0.5)

        # Scale down and move original data to top
        self.play(
            original_data.animate.scale(0.7).to_edge(UP, buff=2.2)
        )

        # Remove group and add individual cards
        self.remove(original_data)
        individual_cards = []
        for card in original_data:
            self.add(card)
            individual_cards.append(card)

        # Define sample positions and create boxes
        n_bags = 3
        sample_positions = [
            np.array([-4, -2.5, 0]),
            np.array([0, -2.5, 0]),
            np.array([4, -2.5, 0])
        ]

        # Create sample boxes
        sample_boxes = []
        for i in range(n_bags):
            box = self.create_sample_box(sample_positions[i], f"Sample {i+1}")
            sample_boxes.append(box)
            self.play(FadeIn(box), run_time=0.4)

        self.wait(0.3)

        # Generate bootstrap samples and animate card throwing
        bootstrap_groups = []
        
        for bag_idx in range(n_bags):
            np.random.seed(40 + bag_idx)
            indices = np.random.choice(len(X), size=15, replace=True)
            
            sample = VGroup()
            should_rotate = (bag_idx != 1)  # Rotate samples 1 and 3, not sample 2
            
            # Calculate grid positions for this sample
            grid_positions = self.calculate_grid_positions(sample_positions[bag_idx])
            
            # Highlight current box
            self.play(
                sample_boxes[bag_idx][0].animate.set_stroke(YELLOW, width=4),
                run_time=0.3
            )
            
            # Throw cards with staggered animation
            if bag_idx == 0:
                # First sample: slow for first 10 cards
                for j in range(10):
                    idx = indices[j]
                    source_card = individual_cards[idx]
                    card_id = source_card.card_id
                    dot_color = source_card.dot_color
                    
                    source_pos = source_card.get_center().copy()
                    
                    card_copy = self.throw_card_manual_trajectory(
                        source_pos, card_id, dot_color, grid_positions[j],
                        should_rotate=should_rotate, run_time=0.8, num_steps=20
                    )
                    sample.add(card_copy)
                
                # Fast dealing for remaining 5 cards
                trajectory_batch = []
                for j in range(10, 15):
                    idx = indices[j]
                    source_card = individual_cards[idx]
                    card_id = source_card.card_id
                    dot_color = source_card.dot_color
                    
                    source_pos = source_card.get_center().copy()
                    
                    traj_data = self.throw_card_manual_fast(
                        source_pos, card_id, dot_color, grid_positions[j],
                        should_rotate=should_rotate, run_time=0.8, num_steps=10
                    )
                    sample.add(traj_data['card'])
                    trajectory_batch.append(traj_data)
                
                self.animate_trajectories_batch(trajectory_batch, run_time=0.8)
            else:
                # Samples 2 and 3: all fast
                trajectory_batch = []
                for j, idx in enumerate(indices):
                    source_card = individual_cards[idx]
                    card_id = source_card.card_id
                    dot_color = source_card.dot_color
                    
                    source_pos = source_card.get_center().copy()
                    
                    traj_data = self.throw_card_manual_fast(
                        source_pos, card_id, dot_color, grid_positions[j],
                        should_rotate=should_rotate, run_time=0.8, num_steps=10
                    )
                    sample.add(traj_data['card'])
                    trajectory_batch.append(traj_data)
                    
                    if len(trajectory_batch) >= 5 or j == len(indices) - 1:
                        self.animate_trajectories_batch(trajectory_batch, run_time=0.6)
                        trajectory_batch = []
            
            # Reset box highlight
            self.play(
                sample_boxes[bag_idx][0].animate.set_stroke(WHITE, width=2),
                run_time=0.2
            )
            
            bootstrap_groups.append((sample_boxes[bag_idx][1], sample_boxes[bag_idx][0], sample))
            self.wait(0.3)

        self.wait(1)

        # Fade out original dataset
        self.play(FadeOut(VGroup(*individual_cards)))
        
        # Fade out sample titles
        for label, _, _ in bootstrap_groups:
            self.play(FadeOut(label), run_time=0.25)

        # Move bootstrap groups up (CHANGED FROM 3 TO 1.5)
        anims = []
        for _, box, group in bootstrap_groups:
            anims.append(box.animate.shift(UP * 3.5))
            anims.append(group.animate.shift(UP * 3.5))
        
        self.play(*anims, run_time=1.2)

        # STEP 3 — TRAIN TREES (under samples)
        new_step = Text("3) Train a decision tree on each bootstrap sample", font_size=26, color=GREEN)
        new_step.move_to(step_text.get_center())
        self.play(Transform(step_text, new_step))

        tree_icons = []
        for i, (_, box, group) in enumerate(bootstrap_groups):
            x_pos = -4 + i * 4
            tree = self.create_simple_tree_icon(scale=0.5)
            tree.move_to(RIGHT * x_pos + DOWN * 1)
            tree_icons.append(tree)
            self.play(FadeIn(tree), run_time=0.6)

        self.wait(0.8)

        # STEP 4 — VOTE ON NEW POINT
        new_step = Text("4) Trees vote on the new point", font_size=26, color=ORANGE)
        new_step.move_to(step_text.get_center())
        self.play(Transform(step_text, new_step))

        new_point = Dot(color=YELLOW, radius=0.14)
        new_point.move_to(DOWN * 2.3)
        self.play(GrowFromCenter(new_point), run_time=0.6)

        # Arrows from trees -> new point
        arrows = VGroup()
        for tree in tree_icons:
            arrows.add(Arrow(tree.get_bottom(), new_point.get_top(), buff=0.15, color=YELLOW))
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2), run_time=0.8)

        # Predictions above trees
        predictions = [RED, BLUE, RED]
        pred_labels = VGroup()
        for tree, c in zip(tree_icons, predictions):
            txt = Text(f"Class {1 if c==RED else 0}", font_size=16, color=c, weight=BOLD)
            txt.next_to(tree, UP, buff=0.15)
            pred_labels.add(txt)
            self.play(Write(txt), run_time=0.35)

        self.wait(0.8)

        # Final pred box under point
        new_step = Text("5) Final majority-vote prediction", font_size=26, color=GREEN)
        new_step.move_to(step_text.get_center())
        self.play(Transform(step_text, new_step))

        red_c = predictions.count(RED)
        blue_c = predictions.count(BLUE)
        final_color = RED if red_c > blue_c else BLUE
        final_class = 1 if final_color == RED else 0

        pred_box = Rectangle(width=4, height=1, color=final_color, stroke_width=3, fill_opacity=0.15)
        pred_box.next_to(new_point, DOWN, buff=0.2)
        pred_text = Text(f"Prediction: Class {final_class}", font_size=26, color=final_color, weight=BOLD)
        pred_text.move_to(pred_box.get_center())

        self.play(Create(pred_box), Write(pred_text), run_time=0.8)
        self.wait(1.0)

        # ===================================================================
        # TRANSITION TO RANDOM FOREST
        # ===================================================================
        self.play(
            Transform(title, Text("Random Forest", font_size=44, weight=BOLD, color=BLUE).to_edge(UP)),
            FadeOut(pred_labels),
            FadeOut(pred_box),
            FadeOut(pred_text),
            FadeOut(step_text),
            run_time=1.0
        )

        slide_up_anims = []
        for _, box, group in bootstrap_groups:    
            slide_up_anims.append(box.animate.shift(UP * 0.5))
            slide_up_anims.append(group.animate.shift(UP * 0.5))
        self.play(*slide_up_anims, run_time=1.0)

        # Colored stripe across the middle
        mid_box = Rectangle(width=config.frame_width*1.9, height=1.2, color=RED, fill_opacity=0.22)
        mid_box.move_to(ORIGIN + DOWN * 1)
        mid_text = Text("Intermediate Step: Feature Selection", font_size=28, weight=BOLD, color=RED)
        mid_text.next_to(mid_box, UP, buff=0.15)
        self.play(FadeIn(mid_box), Write(mid_text), run_time=0.8)
        self.wait(0.5)

        # Highlight feature selection on trees
        highlight_anims = []
        for tree in tree_icons:
            circle = Circle(radius=0.5, color=YELLOW, stroke_width=3)
            circle.move_to(tree.get_center())
            highlight_anims.append(Create(circle))
            self.play(Create(circle), run_time=0.5)
            self.play(FadeOut(circle), run_time=0.3)
        self.wait(0.5)
        self.play(FadeOut(mid_box), FadeOut(mid_text), run_time=0.6)

        # Re-predict using the selected feature subset
        new_predictions = [BLUE, BLUE, RED]
        new_pred_labels = VGroup()
        for tree, c in zip(tree_icons, new_predictions):
            txt = Text(f"Class {1 if c==RED else 0}", font_size=18, color=c, weight=BOLD)
            txt.next_to(tree, UP, buff=0.15)
            new_pred_labels.add(txt)
            self.play(Write(txt), run_time=0.35)

        # Final RF prediction under the yellow point
        red_c2 = new_predictions.count(RED)
        blue_c2 = new_predictions.count(BLUE)
        rf_color = RED if red_c2 > blue_c2 else BLUE
        rf_class = 1 if rf_color == RED else 0

        final_box2 = Rectangle(width=4, height=1, color=rf_color, stroke_width=3, fill_opacity=0.15)
        final_box2.next_to(new_point, DOWN, buff=0.2)
        final_text2 = Text(f"Prediction: Class {rf_class}", font_size=26, color=rf_color, weight=BOLD)
        final_text2.move_to(final_box2.get_center())

        self.play(Create(final_box2), Write(final_text2), run_time=0.8)
        self.wait(2)

    # ========================================================================
    # THEATER ANIMATION METHODS (WITH CARDS)
    # ========================================================================
    
    def create_sample_box(self, position, label_text):
        """Create a box container for a bootstrap sample"""
        card_width = 0.4 * 0.8
        card_height = 0.6 * 0.8
        card_buff = 0.1
        padding = 0.3
        
        box_width = 5 * card_width + 4 * card_buff + 2 * padding
        box_height = 3 * card_height + 2 * card_buff + 2 * padding
        
        box = RoundedRectangle(
            width=box_width,
            height=box_height,
            corner_radius=0.1,
            stroke_color=WHITE,
            stroke_width=2,
            fill_opacity=0.05,
            fill_color=BLUE_E
        )
        box.move_to(position)
        
        label = Text(label_text, font_size=24, color=WHITE)
        label.move_to(box.get_top() + UP * 0.3)
        
        return VGroup(box, label)

    def calculate_grid_positions(self, box_center):
        """Calculate grid positions for cards inside a box"""
        card_width = 0.4 * 0.8
        card_height = 0.6 * 0.8
        card_buff = 0.1
        
        rows = 3
        cols = 5
        
        grid_width = cols * card_width + (cols - 1) * card_buff
        grid_height = rows * card_height + (rows - 1) * card_buff
        
        start_x = box_center[0] - grid_width / 2 + card_width / 2
        start_y = box_center[1] + grid_height / 2 - card_height / 2
        
        positions = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * (card_width + card_buff)
                y = start_y - row * (card_height + card_buff)
                positions.append(np.array([x, y, 0]))
        
        return positions

    def throw_card_manual_trajectory(self, source_pos, card_id, dot_color, target_position, 
                                     should_rotate=True, run_time=0.8, num_steps=20):
        """Manually control card trajectory with explicit positions and angles"""
        
        card_width = 0.4 * 0.8
        card_height = 0.6 * 0.8
        
        # Create card (full card design)
        new_card_rect = RoundedRectangle(
            width=card_width,
            height=card_height,
            corner_radius=0.05,
            fill_opacity=1,
            fill_color=BLUE_E,
            stroke_color=WHITE,
            stroke_width=2
        )
        
        # Add number (positioned lower)
        new_label = Text(str(card_id), font_size=16, color=WHITE)
        new_label.move_to(new_card_rect.get_center() + DOWN * 0.05)
        
        # Add colored dot at top
        new_dot = Dot(radius=0.04, color=dot_color)
        new_dot.move_to(new_card_rect.get_top() + DOWN * 0.12)
        
        card_copy = VGroup(new_card_rect, new_label, new_dot)
        card_copy.card_id = card_id
        card_copy.original_id = card_id
        card_copy.move_to(source_pos)
        
        self.add(card_copy)
        
        pivot_point = card_copy.get_corner(UL).copy()
        current_angle = 0
        
        # Phase 1: Swing back
        swing_angle = 45 * DEGREES
        self.play(
            Rotate(card_copy, angle=swing_angle, about_point=pivot_point),
            run_time=run_time * 0.15,
            rate_func=smooth
        )
        current_angle += swing_angle
        
        # Phase 2: Wind-up
        current_bottom_center = card_copy.get_bottom()
        vec_pivot_to_target = target_position - pivot_point
        desired_direction = np.arctan2(vec_pivot_to_target[1], vec_pivot_to_target[0])
        vec_pivot_to_bottom = current_bottom_center - pivot_point
        current_bottom_angle = np.arctan2(vec_pivot_to_bottom[1], vec_pivot_to_bottom[0])
        angle_to_align = desired_direction - current_bottom_angle
        windup_angle = angle_to_align - 10 * DEGREES
        
        self.play(
            Rotate(card_copy, angle=windup_angle, about_point=pivot_point),
            run_time=run_time * 0.25,
            rate_func=smooth
        )
        current_angle += windup_angle
        
        # Phase 3: Manual trajectory
        start_pos = card_copy.get_center().copy()
        
        if should_rotate:
            normalized_angle = current_angle % (2 * PI)
            total_rotation = 2 * PI - normalized_angle
        else:
            normalized_angle = current_angle % (2 * PI)
            total_rotation = -normalized_angle
        
        trajectory = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            pos = start_pos + alpha * (target_position - start_pos)
            angle = current_angle + alpha * total_rotation
            trajectory.append((pos, angle))
        
        step_time = (run_time * 0.6) / num_steps
        
        for i, (pos, angle) in enumerate(trajectory[1:], 1):
            prev_angle = trajectory[i-1][1]
            angle_delta = angle - prev_angle
            
            card_copy.move_to(pos)
            card_copy.rotate(angle_delta)
            
            if i % 2 == 0 or i == len(trajectory) - 1:
                self.wait(step_time * 2)
        
        return card_copy

    def throw_card_manual_fast(self, source_pos, card_id, dot_color, target_position, 
                              should_rotate=True, run_time=0.8, num_steps=10):
        """Fast version with fewer steps"""
        
        card_width = 0.4 * 0.8
        card_height = 0.6 * 0.8
        
        # Create card
        new_card_rect = RoundedRectangle(
            width=card_width,
            height=card_height,
            corner_radius=0.05,
            fill_opacity=1,
            fill_color=BLUE_E,
            stroke_color=WHITE,
            stroke_width=2
        )
        
        new_label = Text(str(card_id), font_size=16, color=WHITE)
        new_label.move_to(new_card_rect.get_center() + DOWN * 0.05)
        
        new_dot = Dot(radius=0.04, color=dot_color)
        new_dot.move_to(new_card_rect.get_top() + DOWN * 0.12)
        
        card_copy = VGroup(new_card_rect, new_label, new_dot)
        card_copy.card_id = card_id
        card_copy.original_id = card_id
        card_copy.move_to(source_pos)
        
        self.add(card_copy)
        
        pivot_point = card_copy.get_corner(UL).copy()
        
        # Quick swing and windup (no animation)
        swing_angle = 45 * DEGREES
        card_copy.rotate(swing_angle, about_point=pivot_point)
        current_angle = swing_angle
        
        # Calculate windup
        current_bottom_center = card_copy.get_bottom()
        vec_pivot_to_target = target_position - pivot_point
        desired_direction = np.arctan2(vec_pivot_to_target[1], vec_pivot_to_target[0])
        vec_pivot_to_bottom = current_bottom_center - pivot_point
        current_bottom_angle = np.arctan2(vec_pivot_to_bottom[1], vec_pivot_to_bottom[0])
        angle_to_align = desired_direction - current_bottom_angle
        windup_angle = angle_to_align - 10 * DEGREES
        
        card_copy.rotate(windup_angle, about_point=pivot_point)
        current_angle += windup_angle
        
        # Calculate trajectory
        start_pos = card_copy.get_center().copy()
        
        if should_rotate:
            normalized_angle = current_angle % (2 * PI)
            total_rotation = 2 * PI - normalized_angle
        else:
            normalized_angle = current_angle % (2 * PI)
            total_rotation = -normalized_angle
        
        return {
            'card': card_copy,
            'start_pos': start_pos,
            'target_pos': target_position,
            'start_angle': current_angle,
            'total_rotation': total_rotation,
            'num_steps': num_steps
        }

    def animate_trajectories_batch(self, trajectory_data_list, run_time=0.8):
        """Animate multiple card trajectories simultaneously"""
        if not trajectory_data_list:
            return
        
        max_steps = max(data['num_steps'] for data in trajectory_data_list)
        step_time = run_time / max_steps
        
        for step_idx in range(max_steps + 1):
            for data in trajectory_data_list:
                if step_idx <= data['num_steps']:
                    alpha = step_idx / data['num_steps']
                    
                    pos = data['start_pos'] + alpha * (data['target_pos'] - data['start_pos'])
                    target_angle = data['start_angle'] + alpha * data['total_rotation']
                    
                    if step_idx == 0:
                        data['current_angle'] = data['start_angle']
                    
                    angle_delta = target_angle - data['current_angle']
                    
                    data['card'].move_to(pos)
                    data['card'].rotate(angle_delta)
                    
                    data['current_angle'] = target_angle
            
            if step_idx < max_steps:
                self.wait(step_time)

    def create_simple_tree_icon(self, scale=1.0):
        # A simple two-circle tree icon with connecting lines
        root = Circle(radius=0.2 * scale, color=ORANGE, fill_opacity=0.9)
        left = Circle(radius=0.14 * scale, color=BLUE, fill_opacity=0.9)
        right = Circle(radius=0.14 * scale, color=RED, fill_opacity=0.9)
        left.shift(DOWN * 0.35 * scale + LEFT * 0.3 * scale)
        right.shift(DOWN * 0.35 * scale + RIGHT * 0.3 * scale)
        e1 = Line(root.get_bottom(), left.get_top(), stroke_width=2)
        e2 = Line(root.get_bottom(), right.get_top(), stroke_width=2)
        return VGroup(e1, e2, root, left, right)
    

class RandomForestBoundary(Scene):
    def construct(self):
        # Title
        title = Text("Random Forest: Decision Boundary Evolution", 
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
        self.stats_box.to_corner(DL).shift(UP * 0.4 + RIGHT * 0.3)
        
        stats_title = Text("Model Statistics", font_size=20, weight=BOLD, color=GREEN)
        stats_title.next_to(self.stats_box.get_top(), DOWN, buff=0.15)

        n_features = X.shape[1]
        max_features_value = int(np.floor(np.sqrt(n_features)))

        self.stats_content = VGroup(
            Text("Estimators: --", font_size=18),
            Text("Accuracy: --", font_size=18),
            Text(f"Max depth: 5", font_size=16, color=GRAY),
            VGroup(Text("Max features: ", font_size=16, color=GRAY), MathTex(r"\lfloor\sqrt{d}\rfloor", font_size=20)).arrange(RIGHT, buff=0.1)
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
        self.status_text = Text("Building Random Forest model...", font_size=22, color=YELLOW)
        self.status_text.next_to(self.stats_box, UP, buff=1.5)
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
        legend.shift(DOWN * 0.2)
        self.play(FadeIn(legend, scale=0.95), run_time=0.7)
        self.legend = legend
        self.wait(0.3)
        
        # =================================================================
        # Initialize boundary blocks storage
        # =================================================================
        self.boundary_blocks = None
        
        # =================================================================
        # Progressive Random Forest evolution
        # =================================================================
        estimators_list = [1, 2, 5, 10, 20, 50]
        
        for idx, n_estimators in enumerate(estimators_list):
            # Update status text
            status_msg = f"Random Forest with {n_estimators} tree{'s' if n_estimators > 1 else ''}"
            self.update_status_text(status_msg)
            
            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features_value,
                max_depth=5,
                random_state=42,
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
            Text(f"Max depth: 5", font_size=16, color=GRAY),
            Text(f"Max features: 1", font_size=16, color=GRAY)
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
        
        # Get prediction probabilities for confidence
        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        # Get probability of class 1
        Z_prob = Z_proba[:, 1].reshape(xx.shape)
        
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
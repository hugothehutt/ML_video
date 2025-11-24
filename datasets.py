from manim import *
import numpy as np
from sklearn.datasets import make_moons

# ============================================================================
# SCENE: Moon Dataset Introduction (Updated with consistent colors)
# ============================================================================
class MoonDatasetIntro(Scene):
    def construct(self):
        # Title and subtitle
        title = Text("Dataset", font_size=44, weight=BOLD)
        title.to_edge(UP).shift(DOWN * 0.3)
        
        subtitle = Text("Moon-Shaped Clusters", font_size=32, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP), run_time=0.8)
        self.wait(1)
        
        # Generate moon dataset
        np.random.seed(42)
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        
        # Normalize to better range for visualization (0 to 10)
        X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 8 + 1
        
        # Create coordinate axes (smaller and centered better)
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=5.5,
            y_length=5.5,
            axis_config={
                "include_tip": True,
                "tip_width": 0.2,
                "tip_height": 0.2,
                "include_numbers": True,
                "font_size": 20
            }
        )
        axes.shift(DOWN * 0.5)
        
        # Axis labels
        x_label = Text("Feature 1", font_size=20)
        x_label.next_to(axes.x_axis, DOWN, buff=0.1)
        
        y_label = Text("Feature 2", font_size=20)
        y_label.next_to(axes.y_axis, LEFT, buff=0.1)
        y_label.rotate(90 * DEGREES)
        
        # Animate axes creation
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Create data points with PURE_BLUE and PURE_RED consistent styling
        class_0_dots = VGroup()
        class_1_dots = VGroup()
        
        for point, label in zip(X_normalized, y):
            if label == 0:
                dot = Dot(
                    axes.c2p(point[0], point[1]),
                    color=PURE_BLUE,
                    radius=0.07,
                    stroke_width=1.5,
                    stroke_color=WHITE,
                    fill_opacity=1.0
                )
                dot.set_z_index(10)
                class_0_dots.add(dot)
            else:
                dot = Dot(
                    axes.c2p(point[0], point[1]),
                    color=PURE_RED,
                    radius=0.07,
                    stroke_width=1.5,
                    stroke_color=WHITE,
                    fill_opacity=1.0
                )
                dot.set_z_index(10)
                class_1_dots.add(dot)
        
        # Create legend with PURE_BLUE and PURE_RED
        legend_box = Rectangle(
            width=2.5,
            height=1.2,
            color=WHITE,
            stroke_width=2,
            fill_opacity=0.1
        )
        
        class_0_legend = VGroup(
            Dot(color=PURE_BLUE, radius=0.1, stroke_width=1, stroke_color=WHITE),
            Text("Class 0", font_size=24)
        ).arrange(RIGHT, buff=0.3)
        
        class_1_legend = VGroup(
            Dot(color=PURE_RED, radius=0.1, stroke_width=1, stroke_color=WHITE),
            Text("Class 1", font_size=24)
        ).arrange(RIGHT, buff=0.3)
        
        legend_content = VGroup(class_0_legend, class_1_legend)
        legend_content.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        legend_content.move_to(legend_box.get_center())
        
        legend = VGroup(legend_box, legend_content)
        legend.to_corner(UR).shift(LEFT * 0.3 + DOWN * 0.5)
        
        # Animate data points appearing
        self.play(Write(legend), run_time=1)
        self.wait(0.3)
        
        # Show Class 0 points
        self.play(
            LaggedStart(
                *[GrowFromCenter(dot) for dot in class_0_dots],
                lag_ratio=0.02
            ),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Show Class 1 points
        self.play(
            LaggedStart(
                *[GrowFromCenter(dot) for dot in class_1_dots],
                lag_ratio=0.02
            ),
            run_time=1.5
        )
        self.wait(1)
        
        # Add dataset information
        info_text = VGroup(
            Text("200 samples", font_size=24),
            Text("2 features", font_size=24),
            Text("2 classes", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        info_box = SurroundingRectangle(
            info_text,
            color=YELLOW,
            buff=0.3,
            corner_radius=0.1
        )
        
        info_group = VGroup(info_box, info_text)
        info_group.to_corner(UL).shift(RIGHT * 0.3 + DOWN * 0.5)
        
        self.play(
            Create(info_box),
            Write(info_text),
            run_time=1.2
        )
        self.wait(1.5)
        
        # Keep everything on screen for a moment
        self.wait(1)


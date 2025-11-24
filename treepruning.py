from manim import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

# ============================================================================
# SCENE 6: Tree Pruning with Cost-Complexity (FIXED with stable node encoding)
# ============================================================================
class TreePruningScene(Scene):
    def construct(self):
        # Title
        title = Text("Tree Pruning", font_size=44, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # =================================================================
        # PART 1: Show formula explanation in center
        # =================================================================
        formula_box = RoundedRectangle(width=8.5, height=4.5, corner_radius=0.35,
                                       stroke_width=4, color=TEAL, fill_opacity=0.06)
        formula_box.move_to(ORIGIN)

        main_text = Text("Given a maximally grown tree T₀, look for a subtree Tₐ ⊂ T₀ minimizing",
                         font_size=20, color=WHITE)
        main_text.next_to(formula_box.get_top(), DOWN, buff=0.4)

        cost_classification = MathTex(
            r"\sum_{\ell=1}^{|T|} (1 - \hat{p}_{C_\ell}(R_\ell)) + \alpha |T|",
            font_size=32
        )
        cost_classification.next_to(main_text, DOWN, buff=0.6)

        where_text = Text("where", font_size=18, color=WHITE)
        where_text.next_to(cost_classification, DOWN, buff=0.45)

        bullet_points = VGroup(
            MathTex(r"\text{• } C_\ell \text{ is the majority class in } R_\ell \text{ (criterion = misclassification error).}", font_size=22),
            MathTex(r"\text{• } |T| \text{ is the number of terminal leaves in the tree.}", font_size=22),
            MathTex(r"\text{• } \alpha > 0 \text{ is a regularization parameter.}", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        bullet_points.next_to(where_text, DOWN, buff=0.25)

        self.play(FadeIn(formula_box), run_time=0.8)
        self.play(Write(main_text), run_time=1.0)
        self.play(Write(cost_classification), run_time=1.2)
        self.play(Write(where_text), run_time=0.5)
        for bullet in bullet_points:
            self.play(FadeIn(bullet, shift=UP), run_time=0.4)
        self.wait(1.2)

        # =================================================================
        # PART 2: Shrink and move to corner, keeping only classification formula
        # =================================================================
        self.play(
            FadeOut(main_text),
            FadeOut(where_text),
            FadeOut(bullet_points),
            run_time=0.8
        )
        self.wait(0.4)

        small_box = RoundedRectangle(width=2.75, height=0.6, corner_radius=0.12,
                                     stroke_width=3, color=TEAL, fill_opacity=0.04)
        small_box.to_corner(UL).shift(DOWN * 0.5 + RIGHT * 0.3)

        small_formula = MathTex(
            r"\sum_{\ell=1}^{|T|} (1 - \hat{p}_{C_\ell}(R_\ell)) + \alpha |T|",
            font_size=16
        )
        small_formula.move_to(small_box.get_center())

        self.play(
            Transform(formula_box, small_box),
            Transform(cost_classification, small_formula),
            run_time=1.0
        )
        self.wait(0.6)
        self.formula_group = VGroup(formula_box, cost_classification)

        # =================================================================
        # PART 3: Generate and prune trees with different alpha values
        # =================================================================
        np.random.seed(42)
        
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

        subtitle = Text("Pruning with different α values", font_size=32, color=YELLOW)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(subtitle))
        self.wait(0.4)

        self.create_persistent_stats_box()

        alphas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

        previous_tree = None
        previous_visual_group = None
        previous_path_mapping = None
        previous_positions = None

        for idx, alpha in enumerate(alphas):
            tree = DecisionTreeClassifier(criterion='gini', ccp_alpha=alpha, random_state=42)
            tree.fit(X, y)

            accuracy = tree.score(X, y)
            depth = tree.get_depth()
            n_leaves = tree.get_n_leaves()
            n_nodes = tree.tree_.node_count

            if idx == 0:
                # First iteration: draw tree first, then show alpha
                tree_visual, path_mapping, positions = self.create_tree_visual_sklearn(tree, center=ORIGIN + DOWN * 1, target_fraction=0.60)
                self.play(FadeIn(tree_visual, scale=0.95), run_time=1)
                self.update_stats_animated(accuracy, depth, n_leaves, n_nodes)
                
                # Extract alpha from subtitle and transform to equation
                # Create a copy of the alpha symbol from subtitle
                alpha_copy = Text("α", font_size=32, color=YELLOW)
                # Position it where alpha appears in subtitle (approximate)
                alpha_copy.move_to(subtitle.get_center() + RIGHT * 1.3)
                
                # Create the alpha equation
                alpha_text = Text(f"α = {alpha:.3f}", font_size=32, color=YELLOW)
                alpha_text.move_to(subtitle.get_center())
                
                # Fade out subtitle and let alpha stay
                self.play(
                    alpha_copy.animate,
                    FadeOut(subtitle),                     
                    run_time=0.8
                )

                # Animate alpha_copy to equation
                self.play(
                    Unwrite(alpha_copy),
                    TransformFromCopy(alpha_copy, alpha_text),
                    run_time=0.5
                )
                self.alpha_display = alpha_text

                previous_tree = tree
                previous_visual_group = tree_visual
                previous_path_mapping = path_mapping
                previous_positions = positions
            else:
                # Subsequent iterations: update alpha value
                alpha_text = Text(f"α = {alpha:.3f}", font_size=32, color=YELLOW)
                alpha_text.move_to(subtitle.get_center())
                self.play(Transform(self.alpha_display, alpha_text), run_time=0.45)
                
                self.show_pruning_animation(previous_tree, tree,
                                           previous_visual_group, previous_path_mapping, previous_positions,
                                           accuracy, depth, n_leaves, n_nodes)
                previous_tree = tree
                previous_visual_group = self.previous_tree_visual
                previous_path_mapping = self.previous_path_mapping
                previous_positions = self.previous_positions

            self.wait(1.6)

        self.wait(2)

    # -------------------------
    # PERSISTENT STATS
    # -------------------------
    def create_persistent_stats_box(self):
        self.stats_box = Rectangle(width=3.5, height=2.5, color=WHITE, stroke_width=2, fill_opacity=0.05)
        self.stats_box.to_corner(DL).shift(UP * 0.3 + RIGHT * 0.3)

        stats_title = Text("Tree Statistics", font_size=20, weight=BOLD, color=GREEN)
        stats_title.next_to(self.stats_box.get_top(), DOWN, buff=0.15)

        self.stats_content = VGroup(
            Text("Accuracy: --", font_size=20),
            Text("Tree Depth: --", font_size=18),
            Text("Leaf Nodes: --", font_size=18),
            Text("Total Nodes: --", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        self.stats_content.next_to(stats_title, DOWN, buff=0.2)

        self.play(Create(self.stats_box), Write(stats_title))
        self.play(Write(self.stats_content), run_time=0.8)
        self.stats_title = stats_title
        self.wait(0.3)

    def update_stats_animated(self, accuracy, depth, n_leaves, n_nodes):
        new_stats = VGroup(
            Text(f"Accuracy: {accuracy:.1%}", font_size=20, color=GREEN if accuracy > 0.9 else YELLOW),
            Text(f"Tree Depth: {depth}", font_size=18),
            Text(f"Leaf Nodes: {n_leaves}", font_size=18),
            Text(f"Total Nodes: {n_nodes}", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        new_stats.move_to(self.stats_content.get_center())
        self.play(Transform(self.stats_content, new_stats), run_time=0.6)

    # -------------------------
    # KEY FIX: Path-based node identification
    # -------------------------
    def get_node_path(self, tree_, node_id):
        """
        Get the path from root to this node as a tuple of directions.
        Returns: tuple like ('L', 'R', 'L') meaning left->right->left from root
        This creates a stable identifier for nodes across different trees.
        """
        if node_id == 0:
            return ()
        
        # Build path by traversing from root
        path = []
        
        def find_path(current_id, target_id, current_path):
            if current_id == target_id:
                return current_path
            
            left_child = tree_.children_left[current_id]
            right_child = tree_.children_right[current_id]
            
            if left_child != -1:
                result = find_path(left_child, target_id, current_path + ('L',))
                if result is not None:
                    return result
            
            if right_child != -1:
                result = find_path(right_child, target_id, current_path + ('R',))
                if result is not None:
                    return result
            
            return None
        
        return find_path(0, node_id, ())

    def get_node_by_path(self, tree_, path):
        """
        Get node_id by following a path from root.
        Returns -1 if path doesn't exist in this tree.
        """
        if len(path) == 0:
            return 0
        
        current_id = 0
        for direction in path:
            if direction == 'L':
                current_id = tree_.children_left[current_id]
            else:  # 'R'
                current_id = tree_.children_right[current_id]
            
            if current_id == -1:
                return -1
        
        return current_id

    # -------------------------
    # FIXED PRUNING ANIMATION with path-based matching
    # -------------------------
    def show_pruning_animation(self, old_tree, new_tree, old_visual, old_path_mapping, old_positions,
                               accuracy, depth, n_leaves, n_nodes):
        old_tree_ = old_tree.tree_
        new_tree_ = new_tree.tree_

        # Build fresh path-to-node-id mapping for OLD tree
        old_path_to_nodeid = {}
        
        def build_old_paths(node_id, current_path=()):
            if node_id == -1:
                return
            old_path_to_nodeid[current_path] = node_id
            
            left = old_tree_.children_left[node_id]
            right = old_tree_.children_right[node_id]
            if left != -1:
                build_old_paths(left, current_path + ('L',))
            if right != -1:
                build_old_paths(right, current_path + ('R',))
        
        build_old_paths(0)
        
        # Build new tree's path mapping
        new_path_mapping = {}
        
        def build_new_paths(node_id):
            if node_id == -1:
                return
            path = self.get_node_path(new_tree_, node_id)
            new_path_mapping[path] = node_id
            
            left = new_tree_.children_left[node_id]
            right = new_tree_.children_right[node_id]
            if left != -1:
                build_new_paths(left)
            if right != -1:
                build_new_paths(right)
        
        build_new_paths(0)

        # Find which paths exist in old tree but not in new tree (these were pruned)
        old_paths = set(old_path_mapping.keys())
        new_paths = set(new_path_mapping.keys())
        pruned_paths = old_paths - new_paths

        # Find pruning points: paths that exist in new tree but whose children were pruned
        pruning_point_paths = []
        paths_to_prune = []
        
        for path in old_paths:
            if path in new_paths:
                # Check if this node's children were pruned
                # Use the fresh mapping to get the correct node_id from old_tree
                old_node_id = old_path_to_nodeid.get(path)
                if old_node_id is None:
                    continue
                old_left = old_tree_.children_left[old_node_id]
                old_right = old_tree_.children_right[old_node_id]
                
                # Check if children existed in old tree
                has_old_left = old_left != -1
                has_old_right = old_right != -1
                
                if has_old_left or has_old_right:
                    # Check if this node is now a leaf in new tree
                    new_node_id = new_path_mapping[path]
                    new_is_leaf = (new_tree_.children_left[new_node_id] == -1)
                    
                    if new_is_leaf:
                        pruning_point_paths.append(path)
                        # Mark all descendants for pruning
                        left_path = path + ('L',)
                        right_path = path + ('R',)
                        if left_path in old_paths:
                            paths_to_prune.append(left_path)
                            self._mark_descendants_for_pruning(left_path, old_paths, paths_to_prune)
                        if right_path in old_paths:
                            paths_to_prune.append(right_path)
                            self._mark_descendants_for_pruning(right_path, old_paths, paths_to_prune)

        # Nothing to prune
        if len(paths_to_prune) == 0:
            self.update_stats_animated(accuracy, depth, n_leaves, n_nodes)
            self.previous_tree_visual = old_visual
            self.previous_path_mapping = old_path_mapping
            self.previous_positions = old_positions
            return

        # STEP 1: Highlight pruning points
        highlight_anims = []
        for path in pruning_point_paths:
            if path in old_path_mapping and 'shape' in old_path_mapping[path]:
                shape = old_path_mapping[path]['shape']
                highlight_anims.append(shape.animate.set_stroke(YELLOW, width=6))
        if highlight_anims:
            self.play(*highlight_anims, run_time=0.8)
            self.wait(0.25)

        # STEP 2: Highlight pruned subtrees (RED)
        pruned_elements = VGroup()
        for path in paths_to_prune:
            if path in old_path_mapping:
                data = old_path_mapping[path]
                for key in ['shape', 'label', 'left_edge', 'right_edge']:
                    if key in data:
                        pruned_elements.add(data[key])
        if len(pruned_elements) > 0:
            self.play(*[elem.animate.set_color(RED) for elem in pruned_elements], run_time=0.6)
            self.wait(0.2)

        # STEP 3: Cutting lines (create & flash)
        cutting_lines = VGroup()
        for path in pruning_point_paths:
            if path in old_path_mapping:
                data = old_path_mapping[path]
                
                left_path = path + ('L',)
                if left_path in paths_to_prune and 'left_edge' in data:
                    edge = data['left_edge']
                    mid = (edge.get_start() + edge.get_end()) / 2
                    line = Line(mid + LEFT * 0.5, mid + RIGHT * 0.5, color=RED, stroke_width=8)
                    cutting_lines.add(line)

                right_path = path + ('R',)
                if right_path in paths_to_prune and 'right_edge' in data:
                    edge = data['right_edge']
                    mid = (edge.get_start() + edge.get_end()) / 2
                    line = Line(mid + LEFT * 0.5, mid + RIGHT * 0.5, color=RED, stroke_width=8)
                    cutting_lines.add(line)

        if len(cutting_lines) > 0:
            self.play(LaggedStart(*[Create(line) for line in cutting_lines], lag_ratio=0.12), run_time=0.9)
            for _ in range(2):
                self.play(*[line.animate.set_stroke(YELLOW, width=10) for line in cutting_lines], run_time=0.12)
                self.play(*[line.animate.set_stroke(RED, width=8) for line in cutting_lines], run_time=0.12)
            self.wait(0.2)

        # STEP 4: Fade/prune removed subtree nodes & edges
        fade_anims = []
        for path in paths_to_prune:
            if path in old_path_mapping:
                data = old_path_mapping[path]
                for edge_key in ['left_edge', 'right_edge']:
                    if edge_key in data:
                        edge = data[edge_key]
                        try:
                            edge.clear_updaters()
                        except Exception:
                            pass
                        fade_anims.append(edge.animate.set_opacity(0).scale(0.6))
                if 'shape' in data:
                    fade_anims.append(data['shape'].animate.set_opacity(0).scale(0.6))
                if 'label' in data:
                    fade_anims.append(data['label'].animate.set_opacity(0).scale(0.6))

        if fade_anims:
            self.play(*fade_anims, run_time=0.9)
            self.wait(0.12)

        # Remove parent edges to pruned children
        parent_edge_fades = []
        for path in pruning_point_paths:
            if path in old_path_mapping:
                entry = old_path_mapping[path]
                left_path = path + ('L',)
                if 'left_edge' in entry and left_path in paths_to_prune:
                    try:
                        entry['left_edge'].clear_updaters()
                        parent_edge_fades.append(entry['left_edge'].animate.set_opacity(0))
                        del entry['left_edge']
                    except Exception:
                        pass
                right_path = path + ('R',)
                if 'right_edge' in entry and right_path in paths_to_prune:
                    try:
                        entry['right_edge'].clear_updaters()
                        parent_edge_fades.append(entry['right_edge'].animate.set_opacity(0))
                        del entry['right_edge']
                    except Exception:
                        pass
        if parent_edge_fades:
            self.play(*parent_edge_fades, run_time=0.45)
            self.wait(0.08)

        # STEP 5: Replace pruning-point internal nodes with leaf nodes
        replacement_pairs = []
        for path in pruning_point_paths:
            if path not in old_path_mapping:
                continue
            data = old_path_mapping[path]
            if 'shape' not in data:
                continue
            old_shape = data['shape']
            old_label = data.get('label', None)

            # Get info from new tree
            new_node_id = new_path_mapping[path]
            vals = new_tree_.value[new_node_id][0]
            maj_class = int(np.argmax(np.asarray(vals).ravel())) if len(np.asarray(vals).ravel()) > 1 else 0
            n_samples_new = int(new_tree_.n_node_samples[new_node_id])

            cx, cy, _ = old_shape.get_center()
            new_shape, new_label = self.create_leaf_visual(cx, cy, maj_class, n_samples_new, scale=1.0)

            small_scale = 0.8
            new_shape.scale(small_scale)
            new_label.scale(small_scale)
            new_shape.set_opacity(1.0)
            new_label.set_opacity(0.0)

            self.add(new_shape, new_label)

            replacement_pairs.append((path, old_shape, old_label, new_shape, new_label, maj_class, n_samples_new, small_scale))

        # Morph to new leaf
        repl_anims = []
        for (path, old_shape, old_label, new_shape, new_label, majc, ns, sscale) in replacement_pairs:
            repl_anims.append(ReplacementTransform(old_shape, new_shape))
            if old_label is not None:
                repl_anims.append(ReplacementTransform(old_label, new_label))

        if repl_anims:
            self.play(*repl_anims, run_time=0.7)
            self.wait(0.06)

        # STEP 6: Fade-in + grow the newly created leaf(s)
        grow_anims = []
        for (path, old_shape, old_label, new_shape, new_label, majc, ns, sscale) in replacement_pairs:
            grow_anims.append(new_shape.animate.scale(1.0 / sscale))
            grow_anims.append(new_label.animate.set_opacity(1.0).scale(1.0 / sscale))
        if grow_anims:
            self.play(*grow_anims, run_time=0.4)
            self.wait(0.06)

        # Update path mapping entries
        for (path, old_shape, old_label, new_shape, new_label, majc, ns, sscale) in replacement_pairs:
            entry = old_path_mapping.get(path, {})
            entry['shape'] = new_shape
            entry['label'] = new_label
            entry['is_leaf'] = True
            entry['predicted_class'] = majc
            entry['n_samples'] = ns
            entry['node_id'] = new_path_mapping[path]
            if 'left' in entry:
                del entry['left']
            if 'right' in entry:
                del entry['right']

        # STEP 7: Build fresh visual for new tree to get correct positions
        temp_group, temp_map, new_positions_by_path = self.create_tree_visual_sklearn(new_tree, center=ORIGIN, target_fraction=0.60)

        # Move surviving nodes to their new positions
        move_anims = []
        for path in old_path_mapping.keys():
            if path in new_positions_by_path and path not in paths_to_prune:
                target_pos = new_positions_by_path[path]
                data = old_path_mapping[path]
                if 'shape' in data:
                    move_anims.append(data['shape'].animate.move_to(target_pos))
                if 'label' in data:
                    move_anims.append(data['label'].animate.move_to(target_pos))
        if move_anims:
            self.play(*move_anims, run_time=0.9)

        # STEP 8: Cleanup pruned objects
        for path in paths_to_prune:
            if path in old_path_mapping:
                data = old_path_mapping.pop(path)
                for key in ['left_edge', 'right_edge', 'shape', 'label']:
                    if key in data:
                        try:
                            obj = data[key]
                            try:
                                obj.clear_updaters()
                            except Exception:
                                pass
                            if obj in self.mobjects:
                                self.remove(obj)
                        except Exception:
                            pass

        # STEP 9: Fade out cutting lines
        if len(cutting_lines) > 0:
            self.play(*[line.animate.set_opacity(0) for line in cutting_lines], run_time=0.4)
            for line in cutting_lines:
                try:
                    self.remove(line)
                except Exception:
                    pass

        # STEP 10: Final cleanup
        for path, data in list(old_path_mapping.items()):
            for edge_key in ['left_edge', 'right_edge']:
                if edge_key in data:
                    edge = data[edge_key]
                    try:
                        if edge.get_opacity() == 0:
                            edge.clear_updaters()
                            if edge in self.mobjects:
                                self.remove(edge)
                    except Exception:
                        pass

        # STEP 11: Update stats
        self.update_stats_animated(accuracy, depth, n_leaves, n_nodes)

        # STEP 12: Build current group
        current_group = VGroup()
        for path, data in old_path_mapping.items():
            if 'shape' in data:
                current_group.add(data['shape'])
            if 'label' in data:
                current_group.add(data['label'])
            if 'left_edge' in data:
                current_group.add(data['left_edge'])
            if 'right_edge' in data:
                current_group.add(data['right_edge'])

        # Store for next iteration
        self.previous_tree_visual = current_group
        self.previous_path_mapping = old_path_mapping
        self.previous_positions = new_positions_by_path

    def _mark_descendants_for_pruning(self, path, all_paths, prune_list):
        """Recursively mark all descendants of a path for pruning."""
        left_path = path + ('L',)
        right_path = path + ('R',)
        
        if left_path in all_paths:
            prune_list.append(left_path)
            self._mark_descendants_for_pruning(left_path, all_paths, prune_list)
        
        if right_path in all_paths:
            prune_list.append(right_path)
            self._mark_descendants_for_pruning(right_path, all_paths, prune_list)

    # -------------------------
    # Leaf factory & create visuals (MODIFIED to return path-based mapping)
    # -------------------------
    def create_leaf_visual(self, x, y, predicted_class, n_samples, scale=1.0):
        """Return a consistent leaf shape and label (centered at x,y)."""
        color = BLUE if predicted_class == 0 else RED

        base_radius = 0.35
        radius = base_radius * (scale if scale > 0 else 1.0)

        shape = Circle(
            radius=radius,
            color=color,
            fill_opacity=0.7,
            stroke_width=2,
            stroke_color=WHITE
        )
        shape.move_to([x, y, 0])

        label = VGroup(
            Text(f"C{predicted_class}", font_size=int(14 * (radius / base_radius)), weight=BOLD, color=WHITE),
            Text(f"n={n_samples}", font_size=int(10 * (radius / base_radius)), color=WHITE)
        ).arrange(DOWN, buff=0.05 * (radius / base_radius))
        label.move_to(shape.get_center())

        return shape, label

    def create_tree_visual_sklearn(self, tree, center=ORIGIN, target_fraction=0.60):
        """
        Build visual from sklearn tree; returns:
          - group (VGroup),
          - path_mapping dict (path_tuple -> elements dict),
          - positions dict (path_tuple -> np.array([x,y,0]))
        Uses path-based identification instead of node IDs for stability across trees.
        """
        tree_ = tree.tree_
        n_leaves = tree.get_n_leaves()
        initial_width = max(1.4, n_leaves * 0.35)

        all_elements = VGroup()
        path_mapping = {}
        positions = {}

        def build_node(node_id, x, y, width, depth=0, scale=1.0, path=()):
            if node_id == -1:
                return
            
            is_leaf = (tree_.children_left[node_id] == tree_.children_right[node_id])

            if is_leaf:
                values = tree_.value[node_id][0]
                predicted_class = int(np.argmax(np.asarray(values).ravel())) if len(np.asarray(values).ravel()) > 1 else 0
                n_samples = int(tree_.n_node_samples[node_id])

                shape, label = self.create_leaf_visual(x, y, predicted_class, n_samples, scale=scale)
                all_elements.add(shape, label)
                path_mapping[path] = {
                    'shape': shape,
                    'label': label,
                    'is_leaf': True,
                    'predicted_class': predicted_class,
                    'n_samples': n_samples,
                    'node_id': node_id,
                    'left': -1,
                    'right': -1
                }

            else:
                feature = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                n_samples = int(tree_.n_node_samples[node_id])

                rect = RoundedRectangle(width=1.2 * scale, height=0.7 * scale, corner_radius=0.08 * scale,
                                         color=ORANGE, fill_opacity=0.45, stroke_width=2)
                rect.move_to([x, y, 0])

                label = VGroup(
                    Text(f"X{feature} ≤ {threshold:.2f}", font_size=int(12 * scale), weight=BOLD),
                    Text(f"n={n_samples}", font_size=int(10 * scale))
                ).arrange(DOWN, buff=0.05 * scale)
                label.move_to(rect.get_center())

                all_elements.add(rect, label)
                path_mapping[path] = {
                    'shape': rect,
                    'label': label,
                    'is_leaf': False,
                    'predicted_class': None,
                    'n_samples': n_samples,
                    'node_id': node_id,
                    'left': None,
                    'right': None
                }

                new_width = width * 0.5
                child_y = y - 1.0 * scale
                left_x = x - width * scale * 0.8
                right_x = x + width * scale * 0.8

                left_edge = Line([x, y - 0.35 * scale, 0], [left_x, child_y + 0.35 * scale, 0],
                                 color=BLUE, stroke_width=2 * scale)
                right_edge = Line([x, y - 0.35 * scale, 0], [right_x, child_y + 0.35 * scale, 0],
                                  color=RED, stroke_width=2 * scale)
                all_elements.add(left_edge, right_edge)
                path_mapping[path]['left_edge'] = left_edge
                path_mapping[path]['right_edge'] = right_edge

                left_child = tree_.children_left[node_id]
                right_child = tree_.children_right[node_id]
                path_mapping[path]['left'] = left_child
                path_mapping[path]['right'] = right_child

                # Build children with path extensions
                left_path = path + ('L',)
                right_path = path + ('R',)
                build_node(left_child, left_x, child_y, new_width, depth + 1, scale, left_path)
                build_node(right_child, right_x, child_y, new_width, depth + 1, scale, right_path)

                # Updaters using path-based lookup
                def make_left_updater(edge, parent_path=path, child_path=left_path):
                    def updater(e):
                        try:
                            parent_shape = path_mapping[parent_path]['shape']
                            child_shape = path_mapping[child_path]['shape']
                            e.put_start_and_end_on(parent_shape.get_bottom(), child_shape.get_top())
                        except Exception:
                            pass
                    return updater

                def make_right_updater(edge, parent_path=path, child_path=right_path):
                    def updater(e):
                        try:
                            parent_shape = path_mapping[parent_path]['shape']
                            child_shape = path_mapping[child_path]['shape']
                            e.put_start_and_end_on(parent_shape.get_bottom(), child_shape.get_top())
                        except Exception:
                            pass
                    return updater

                left_edge.add_updater(make_left_updater(left_edge))
                right_edge.add_updater(make_right_updater(right_edge))

        # Build from root with empty path
        build_node(0, 0, 0, initial_width, depth=0, scale=1.0, path=())

        # Scale and center
        current_height = all_elements.height if all_elements.height != 0 else 1.0
        target_height = config.frame_height * target_fraction
        scale_factor = target_height / current_height if current_height > 0 else 1.0
        all_elements.scale(scale_factor)
        all_elements.move_to(center)

        # Record final positions by path
        for path, data in path_mapping.items():
            if 'shape' in data:
                positions[path] = np.array(data['shape'].get_center())

        return all_elements, path_mapping, positions

# End of file
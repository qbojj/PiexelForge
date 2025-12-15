"""Visualization utilities for rasterizer output"""

from typing import List, Tuple


class Fragment:
    """Represents a rasterized fragment"""

    def __init__(
        self, coord_pos: Tuple[int, int], color: Tuple[float, float, float, float]
    ):
        self.coord_pos = coord_pos  # (x, y) position in framebuffer
        self.color = color  # (r, g, b, a) color values in [0, 1]

    def __repr__(self):
        return f"Fragment(pos={self.coord_pos}, color={self.color})"


class FragmentVisualizer:
    """Visualize rasterized fragments in various formats"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def visualize_ascii(self, fragments: List[Fragment], intensity_char=True) -> str:
        """Generate ASCII art visualization of fragments

        Args:
            fragments: List of fragment objects with coord_pos and color
            intensity_char: If True, use intensity-based characters; else solid block

        Returns:
            String representation of the rasterized image
        """
        # Create canvas
        canvas = [["." for _ in range(self.width)] for _ in range(self.height)]

        # Plot fragments
        for frag in fragments:
            x, y = int(frag.coord_pos[0]), int(frag.coord_pos[1])

            # Bounds check
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            # Get color
            r, g, b, a = frag.color
            intensity = (r + g + b) / 3.0  # Average intensity

            if intensity_char:
                if intensity > 0.9:
                    char = "█"  # Full block
                elif intensity > 0.7:
                    char = "▓"  # Dark shade (75%)
                elif intensity > 0.5:
                    char = "▒"  # Medium shade (50%)
                elif intensity > 0.25:
                    char = "░"  # Light shade (25%)
                else:
                    char = "·"  # Light dot
            else:
                char = "█"  # Always solid block

            canvas[y][x] = char

        # Convert to string
        output = []
        for row in canvas:
            output.append("".join(row))
        return "\n".join(output)

    def visualize_color_ascii(self, fragments: List[Fragment]) -> str:
        """Generate colorized ASCII visualization (ANSI codes)

        Args:
            fragments: List of fragment objects

        Returns:
            String with ANSI color codes
        """
        canvas = [
            [[255, 255, 255] for _ in range(self.width)] for _ in range(self.height)
        ]

        # Plot fragments (white background by default)
        for frag in fragments:
            x, y = int(frag.coord_pos[0]), int(frag.coord_pos[1])

            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            r, g, b, a = frag.color
            # Convert from [0, 1] to [0, 255]
            canvas[y][x] = [int(r * 255), int(g * 255), int(b * 255)]

        # Build ANSI string with 24-bit true color
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                r, g, b = canvas[y][x]
                # ANSI 24-bit color escape sequence
                line += f"\033[38;2;{r};{g};{b}m█\033[0m"
            lines.append(line)

        return "\n".join(lines)

    def generate_ppm_image(
        self, fragments: List[Fragment], filepath: str = "rasterizer_output.ppm"
    ):
        """Generate PPM (Portable PixMap) image file

        Args:
            fragments: List of fragment objects
            filepath: Output file path
        """
        # Create image buffer (white background)
        image = [
            [[255, 255, 255] for _ in range(self.width)] for _ in range(self.height)
        ]

        # Plot fragments
        for frag in fragments:
            x, y = int(frag.coord_pos[0]), int(frag.coord_pos[1])

            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            r, g, b, a = frag.color
            # Convert from [0, 1] to [0, 255]
            image[y][x] = [int(r * 255), int(g * 255), int(b * 255)]

        # Write PPM file
        with open(filepath, "w") as f:
            # PPM header
            f.write("P3\n")
            f.write(f"{self.width} {self.height}\n")
            f.write("255\n")

            # Pixel data
            for y in range(self.height):
                row_data = []
                for x in range(self.width):
                    r, g, b = image[y][x]
                    row_data.append(f"{r} {g} {b}")
                f.write(" ".join(row_data) + "\n")

        print(f"Generated PPM image: {filepath}")

    def generate_statistics(self, fragments: List[Fragment]) -> dict:
        """Generate statistics about the rasterized output

        Args:
            fragments: List of fragment objects

        Returns:
            Dictionary with statistics
        """
        if not fragments:
            return {"fragment_count": 0, "coverage": 0.0, "color_ranges": {}}

        color_sum = [0.0, 0.0, 0.0, 0.0]
        min_color = [1.0, 1.0, 1.0, 1.0]
        max_color = [0.0, 0.0, 0.0, 0.0]

        for frag in fragments:
            for i in range(4):
                c = frag.color[i]
                color_sum[i] += c
                min_color[i] = min(min_color[i], c)
                max_color[i] = max(max_color[i], c)

        avg_color = [s / len(fragments) for s in color_sum]

        return {
            "fragment_count": len(fragments),
            "coverage": 100.0 * len(fragments) / (self.width * self.height),
            "average_color": {
                "r": avg_color[0],
                "g": avg_color[1],
                "b": avg_color[2],
                "a": avg_color[3],
            },
            "min_color": {
                "r": min_color[0],
                "g": min_color[1],
                "b": min_color[2],
                "a": min_color[3],
            },
            "max_color": {
                "r": max_color[0],
                "g": max_color[1],
                "b": max_color[2],
                "a": max_color[3],
            },
        }


def print_fragment_summary(fragments: List, visualizer: FragmentVisualizer):
    """Print a comprehensive summary of fragments with visualization

    Args:
        fragments: List of fragment objects
        visualizer: FragmentVisualizer instance
    """
    stats = visualizer.generate_statistics(fragments)

    print("\n" + "=" * 60)
    print("RASTERIZER OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Fragment Count: {stats['fragment_count']}")
    print(f"Coverage: {stats['coverage']:.2f}%")
    print("\nAverage Color (RGBA):")
    for component, value in stats["average_color"].items():
        print(f"  {component.upper()}: {value:.4f}")
    print("\nColor Ranges:")
    print(f"  R: {stats['min_color']['r']:.4f} -> {stats['max_color']['r']:.4f}")
    print(f"  G: {stats['min_color']['g']:.4f} -> {stats['max_color']['g']:.4f}")
    print(f"  B: {stats['min_color']['b']:.4f} -> {stats['max_color']['b']:.4f}")
    print(f"  A: {stats['min_color']['a']:.4f} -> {stats['max_color']['a']:.4f}")

    print("\nASCII Visualization (Intensity-based):")
    print("(█ = high intensity, ░ = medium, · = low)")
    print("-" * visualizer.width)
    print(visualizer.visualize_ascii(fragments, intensity_char=True))
    print("-" * visualizer.width)
    print("=" * 60 + "\n")

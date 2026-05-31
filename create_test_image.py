from PIL import Image, ImageDraw

size = 400
image = Image.new("RGB", (size, size), "white")
draw = ImageDraw.Draw(image)

center = size // 2
radius = 80
draw.ellipse(
    (center - radius, center - radius, center + radius, center + radius),
    fill="red",
)

image.save("test_image.jpg")

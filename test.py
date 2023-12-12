import pyglet


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(MyWindow, self).__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()
        self.point = self.batch.add(3, pyglet.gl.GL_TRIANGLES, None, ('v2f', (0, 0, 100, 0, 50, 100)))


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window")
    pyglet.app.run()

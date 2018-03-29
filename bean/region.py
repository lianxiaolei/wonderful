# coding: utf-8

from collections import OrderedDict

class Region(object):

    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.sub_regions = OrderedDict()
        print 'Class has been initialized', vars(self)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.width = height

    def get_sub_regions(self):
        return self.sub_regions

    def add_sub_region(self, region):
        self.sub_regions.append(region)


if __name__ == '__main__':
    r = Region(10, 10, 2, 2)

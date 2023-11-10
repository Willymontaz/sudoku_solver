#!/usr/bin/python3

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle

from solver import *
from sudoku import *

button_txt_grid_color = (0, 0, 0, 1)
button_txt_nums_color = (1, 1, 1, 1)
button_txt_acts_color = (1, 1, 1, 1)

button_bg_click_color = (1, .7, 0, 1)
buttor_bg_error_color = (1, 0, 0, 1)
button_bg_grid_color = (1, 1, 1, 1)
button_bg_nums_color = (90 / 255, 100 / 255, 240 / 255, .8)
button_bg_acts_color = (90 / 255, 100 / 255, 240 / 255, 1)


class root_layout(FloatLayout):
    def __init__(self, **kwargs):
        super(root_layout, self).__init__(**kwargs)
        self.input_value = None

        # Add grid
        self.grid = grid_layout()
        self.grid.size_hint = None, None
        self.grid_padding = .02
        self.add_widget(self.grid)

        # Add inputs
        self.input_nums = input_numbers()
        self.input_nums.size_hint = None, None
        self.add_widget(self.input_nums)

        self.input_acts = input_actions()
        self.input_acts.size_hint = None, None
        self.add_widget(self.input_acts)

        # Popup
        x, y = Window.size
        self.popup = Popup(
            title='',
            separator_height=0,
            content=Label(text='Processing, please wait...', font_size=self.width * .5),
            size_hint=(None, None),
            size=(x, y)
        )

    def do_layout(self, *args):
        self.apply_ratio()
        super(root_layout, self).do_layout()

    def apply_ratio(self):
        # Resize popup
        self.popup.size = (self.width, self.height)
        self.popup.font_size = self.width * .5

        # Resize elements
        if self.width / self.height > (11 + self.grid_padding * 3) / 9:
            # Calc case size
            case_size = self.height * (1 - self.grid_padding * 2) / 9

            # Set size
            self.grid.size = case_size * 9, case_size * 9

            self.input_nums.cols = 2
            nums_width = self.grid.width / 9 * 2
            nums_height = nums_width * 5 / 2
            self.input_nums.size = nums_width, nums_height

            self.input_acts.cols = 1
            acts_width = nums_width
            acts_height = acts_width * 2
            self.input_acts.size = acts_width, acts_height

            # Set position
            pos_width_gap = ((self.width - self.grid.width - nums_width) / 3) / self.width
            pos_height_gap = ((self.height - self.grid.height) / 2) / self.height
            self.grid.pos_hint = {'x': pos_width_gap, 'y': pos_height_gap}
            self.input_nums.pos_hint = {'x': 2 * pos_width_gap + self.grid.width / self.width,
                                        'y': pos_height_gap + self.input_acts.height / self.height}
            self.input_acts.pos_hint = {'x': 2 * pos_width_gap + self.grid.width / self.width, 'y': pos_height_gap}

        elif self.height / self.width > (11 + self.grid_padding * 3) / 9:
            # Calc case size
            case_size = self.width * (1 - self.grid_padding * 2) / 9

            # Set size
            self.grid.size = case_size * 9, case_size * 9

            self.input_nums.cols = 5
            nums_width = self.grid.width / 9 * 5
            nums_height = self.grid.height / 9 * 2
            self.input_nums.size = nums_width, nums_height

            self.input_acts.cols = 2
            acts_width = nums_height * 2
            acts_height = nums_height
            self.input_acts.size = acts_width, acts_height

            # Set position
            pos_width_gap = ((self.width - self.grid.width) / 2) / self.width
            pos_height_gap = ((self.height - self.grid.height - nums_height) / 3) / self.height
            self.grid.pos_hint = {'x': pos_width_gap, 'y': 2 * pos_height_gap + self.input_nums.height / self.height}
            self.input_nums.pos_hint = {'x': pos_width_gap, 'y': pos_height_gap}
            self.input_acts.pos_hint = {'x': pos_width_gap + self.input_nums.width / self.width, 'y': pos_height_gap}

        elif self.width > self.height:
            # Calc case size
            case_size = self.width * (1 - self.grid_padding * 3) / 11

            # Set size
            self.grid.size = case_size * 9, case_size * 9

            self.input_nums.cols = 2
            nums_width = self.grid.width / 9 * 2
            nums_height = nums_width * 5 / 2
            self.input_nums.size = nums_width, nums_height

            self.input_acts.cols = 1
            acts_width = nums_width
            acts_height = acts_width * 2
            self.input_acts.size = acts_width, acts_height

            # Set position
            pos_width_gap = ((self.width - self.grid.width - nums_width) / 3) / self.width
            pos_height_gap = ((self.height - self.grid.height) / 2) / self.height
            self.grid.pos_hint = {'x': pos_width_gap, 'y': pos_height_gap}
            self.input_nums.pos_hint = {'x': 2 * pos_width_gap + self.grid.width / self.width,
                                        'y': pos_height_gap + self.input_acts.height / self.height}
            self.input_acts.pos_hint = {'x': 2 * pos_width_gap + self.grid.width / self.width, 'y': pos_height_gap}

        else:
            # Calc case size
            case_size = self.height * (1 - self.grid_padding * 3) / 11

            # Set size
            self.grid.size = case_size * 9, case_size * 9

            self.input_nums.cols = 5
            nums_width = self.grid.width / 9 * 5
            nums_height = self.grid.height / 9 * 2
            self.input_nums.size = nums_width, nums_height

            self.input_acts.cols = 2
            acts_width = nums_height * 2
            acts_height = nums_height
            self.input_acts.size = acts_width, acts_height

            # Set position
            pos_width_gap = ((self.width - self.grid.width) / 2) / self.width
            pos_height_gap = ((self.height - self.grid.height - nums_height) / 3) / self.height
            self.grid.pos_hint = {'x': pos_width_gap, 'y': 2 * pos_height_gap + self.input_nums.height / self.height}
            self.input_nums.pos_hint = {'x': pos_width_gap, 'y': pos_height_gap}
            self.input_acts.pos_hint = {'x': pos_width_gap + self.input_nums.width / self.width, 'y': pos_height_gap}

        # Resize fonts
        for x in range(9):
            for y in range(9):
                self.grid.case_layout[x][y].font_size = case_size * .6
                self.grid.case_layout[x][y].width = case_size - 3
                self.grid.case_layout[x][y].height = case_size - 3

        for x in range(10):
            self.input_nums.nums_dict[x].font_size = case_size * .6

        for x in range(2):
            self.input_acts.actions_dict[x].font_size = case_size * .6


class input_actions(GridLayout):
    def __init__(self, **kwargs):
        super(input_actions, self).__init__(**kwargs)
        self.cols = 1

        self.actions_dict = dict()
        for x in range(2):
            if x == 0:
                text_value = "Solve"
            else:
                text_value = "Clear"

            self.actions_dict[x] = Button(
                text=text_value,
                color=button_txt_acts_color,
                background_normal="",
                background_down="",
                background_color=button_bg_acts_color,
                bold=True,
                font_size=self.width / 3
            )
            self.actions_dict[x].value = text_value

            self.add_widget(self.actions_dict[x])
            self.actions_dict[x].bind(on_press=self.button_pressed)
            self.actions_dict[x].bind(on_release=self.button_released)

    def button_pressed(self, instance):
        self.parent.grid.clear_buttons()
        instance.background_color = button_bg_click_color
        if instance.value == "Clear":
            for x in range(9):
                for y in range(9):
                    self.parent.grid.case_layout[x][y].text = ""
                    self.parent.grid.case_layout[x][y].background_color = button_bg_grid_color

            self.parent.input_value = None
            for x in range(10):
                self.parent.input_nums.nums_dict[x].background_color = button_bg_nums_color

        if instance.value == "Solve":
            # Open popup
            self.parent.popup.open()

            # Solve Sudoku
            Clock.schedule_once(lambda dt: self.sudoku_solver_schedule(), 0)

            self.parent.popup.dismiss()

    def sudoku_solver_schedule(self, *args):
        # Create sudoku
        sudoku_obj = sudoku()
        for x in range(9):
            for y in range(9):
                case_value = self.parent.grid.case_layout[x][y].text
                if case_value == "":
                    sudoku_obj.set_value(0, (x, y))
                else:
                    sudoku_obj.set_value(int(case_value), (x, y))

        valid_grid, tuple_list = sudoku_obj.is_valid()

        # Run solver
        if valid_grid:
            # Solve sudoku
            sudoku_solver(sudoku_obj, 10, False)

            # Update Grid
            for x in range(9):
                for y in range(9):
                    if sudoku_obj.get_value((x, y)) != 0:
                        self.parent.grid.case_layout[x][y].text = str(sudoku_obj.get_value((x, y)))

        # Error
        else:
            for tmp_tuple in tuple_list:
                self.parent.grid.case_layout[tmp_tuple[0]][tmp_tuple[1]].background_color = buttor_bg_error_color

    def button_released(self, instance):
        instance.background_color = button_bg_acts_color


class input_numbers(GridLayout):
    def __init__(self, **kwargs):
        super(input_numbers, self).__init__(**kwargs)
        self.cols = 5
        self.padding = 0
        self.spacing = 0

        self.nums_dict = dict()
        for x in range(10):
            if x == 0:
                text_value = ""
            else:
                text_value = str(x)

            self.nums_dict[x] = Button(
                text=text_value,
                color=button_txt_nums_color,
                background_normal="",
                background_down="",
                background_color=button_bg_nums_color,
                bold=True,
                font_size=self.width / 3
            )
            self.nums_dict[x].value = text_value
            self.nums_dict[x].bind(on_press=self.button_pressed)
            self.add_widget(self.nums_dict[x])

    def button_pressed(self, instance):
        # self.parent.grid.clear_buttons()

        # Clear button color            
        for x in range(10):
            self.nums_dict[x].background_color = button_bg_nums_color

        if self.parent.input_value != instance.value:
            self.parent.input_value = instance.value
            # Color button
            instance.background_color = button_bg_click_color
        else:
            self.parent.input_value = None


class grid_layout(GridLayout):
    def __init__(self, **kwargs):
        super(grid_layout, self).__init__(**kwargs)
        self.cols = 3
        self.padding = 0
        self.spacing = 0

        with self.canvas.before:
            Color(0, 0, 0, 1)
            self.rect = Rectangle(size=self.size,
                                  pos=self.pos)
        self.bind(pos=self.update_rect, size=self.update_rect)

        self.width = self.minimum_width
        self.height = self.minimum_height

        # Create Block
        self.block_layout = dict()
        for line_block in range(3):
            self.block_layout[line_block] = dict()
            for row_block in range(3):
                self.block_layout[line_block][row_block] = GridLayout(cols=3, padding=3, spacing=3)
                self.block_layout[line_block][row_block].pos_hint = {'center_x': 0.5, 'center_y': 0.5}
                self.add_widget(self.block_layout[line_block][row_block])

        # Create cases
        self.case_layout = dict()
        for line_case in range(9):
            self.case_layout[line_case] = dict()
            for row_case in range(9):
                self.case_layout[line_case][row_case] = Button(
                    text='',
                    color=button_txt_grid_color,
                    background_normal="",
                    background_down="",
                    background_color=button_bg_grid_color,
                    bold=True,
                    font_size=self.width / 3
                )
                self.case_layout[line_case][row_case].position = (line_case, row_case)
                self.case_layout[line_case][row_case].pressed = False
                self.case_layout[line_case][row_case].bind(on_press=self.button_pressed)
                self.case_layout[line_case][row_case].bind(on_release=self.button_released)

                # Add widget
                self.block_layout[int(line_case / 3)][int(row_case / 3)].add_widget(
                    self.case_layout[line_case][row_case])

    def clear_buttons(self):
        for line_case in range(9):
            for row_case in range(9):
                self.case_layout[line_case][row_case].background_color = button_bg_grid_color

    def update_rect(self, instance, value):
        self.rect.pos = list(map(lambda x: x - 1, self.pos))
        self.rect.size = list(map(lambda x: x + 2, self.size))

    def button_pressed(self, instance):
        self.clear_buttons()
        instance.background_color = button_bg_click_color
        x, y = instance.position
        if self.parent.input_value != None:
            self.case_layout[x][y].text = self.parent.input_value

    def button_released(self, instance):
        instance.background_color = button_bg_grid_color


class SudokuSolverApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        self.root = root_layout()

        return self.root


if __name__ == '__main__':
    SudokuSolverApp().run()

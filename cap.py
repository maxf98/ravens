import os
from time import sleep 
import re
import random
import shapely
import itertools
import traceback
import copy

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer

import base64
from PIL import Image
from io import BytesIO
import ast
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
from scipy.spatial import distance

import openai
from environments.environment import Environment
from dataset import RavensDataset
from utils import utils
import tasks



max_token = 2048
answer = ''
mem = (0, [0], 32)

api_key = "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"
client = openai.OpenAI(api_key=api_key)


class LMP:
    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg[0]
        self._llm = cfg[1]

        self._base_prompt = self._cfg['prompt_text']

        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self.exec_hist = ''

        self.mem = update_memory()

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (f"from utils import "
                                         f"{', '.join(self._variable_vars.keys())}")
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(
            self,
            query,
            context='',
            **kwargs
    ):
        prompt, use_query = self.build_prompt(query, context=context)
        while True:
            try:
                response = client.chat.completions.create(
                        messages=[{"role": "system",
                                   "content": "You are a task planning assistant "
                                              "who only answers with python code"},
                                  {"role": "user",
                                   "content": prompt}],
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )
                code_str = response.choices[0].message.content
                print(response)
                break

            except (openai.APIError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        res = code_str

        if '```' in code_str:
            code_str = extract_code(code_str)
        
        if self._cfg['include_context'] and context != '' and context not in code_str:
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')
        global answer
        answer += f'LLM answer:\n{res}\nLMP {self._name} exec:\n\n{to_log}\n'

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            # print(lvars)
            if self._cfg['return_val_name'] == 'whole_answer':
                return to_exec
            else:
                return lvars[self._cfg['return_val_name']]
        
class LMPV:
    def __init__(self, name, cfg):
        self._name = name
        self._cfg = cfg[0]
        self._llm = cfg[1]

        self._base_prompt = self._cfg['prompt_text']

        self.exec_hist = ''

        self.mem = update_memory()

    def clear_exec_hist(self):
        self.exec_hist = ''

    def encode_image(self, image_sources):
        images = []
        for image_source in image_sources:
            image = Image.fromarray(image_source[0])
            if image.mode == 'F':
                if image_source[1] == 'd':
                    image = image.convert('L')
                elif image_source[1] == 'c':
                    image = image.convert('RGB')
            buffered = BytesIO()
            image.save(buffered, format='JPEG')
            img_str = buffered.getvalue()
            images.append(base64.b64encode(img_str).decode('utf-8'))
        return images

    def build_prompt(self, query, context=''):
        variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(
            self,
            query,
            context=None,
            **kwargs
    ):
        prompt, use_query = self.build_prompt(query)
        images = self.encode_image(context)
        while True:
            try:
                if 'gpt4' in self._llm:
                    code_str = client.chat.completions.create(
                        messages=[{"role": "system",
                                   "content": "You are a task completion checking assistant "
                                              "who compares the final observation with initial observation "
                                              "and gives the judge in python code format like "
                                              "'judge = True (or False)'"
                                              },
                                  {"role": "user",
                                   "content": [
                                       {"type": "text", "text": prompt},

                                       {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[0]}",
                                            },
                                        },
                                        {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[1]}",
                                            },
                                        },
                                        {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[2]}",
                                            },
                                        },
                                        {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[3]}",
                                            },
                                        },
                                    ],
                                },
                            ],
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )['choices'][0]['message']['content'].strip()
                    break
                else:
                    break

            except (openai.APIError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        res = code_str

        if '```' in code_str:
            code_str = extract_code(code_str)

        to_exec = code_str
        to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')
        global answer
        answer += f'LLM answer:\n{res}\nLMP {self._name} exec:\n\n{to_log}\n'

        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['has_return']:
            # print(lvars)
            try:
                return lvars[self._cfg['return_val_name']]
            except:
                if 'True' in to_exec:
                    return True
                else:
                    return False


class LMPFGen:

    def __init__(
            self,
            cfg,
            fixed_vars,
            variable_vars,
            offline_model=None,
            offline_tokenizer=None,
            use_vllm: bool = False
    ):
        self._cfg = cfg[0]
        self._llm = cfg[1]

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

        self.mem = update_memory()

        self.offline_model = offline_model
        self.offline_tokenizer = offline_tokenizer
        self.use_vllm = use_vllm

    def create_f_from_sig(
            self,
            f_name,
            f_sig,
            other_vars=None,
            return_src=False,
    ):
        print(f'Creating function: {f_sig}')

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f'{self._base_prompt}\n{use_query}'
        while True:
            try:
                f_src = client.chat.completions.create(
                        messages=[{"role": "system",
                                   "content": "You are a task planning assistant "
                                              "who only answers with python code"},
                                  {"role": "user",
                                   "content": prompt}],
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    ).choices[0].message.content.strip()
                break

            except (openai.APIError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        res = f_src

        if '```' in f_src:
            f_src = extract_code(f_src)

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')
        global answer
        answer += f'LLM answer:\n{res}\nLMP FGEN created:\n\n{use_query}\n{f_src}\n'

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        if '```' in code_str:
            code_str = extract_code(code_str)

        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(
                    f_name,
                    f_sig,
                    new_fs,
                    fix_bugs=fix_bugs,
                    return_src=True
                )

                # recursively define child_fs in the function body if needed
                f_def_body = ast.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts(
                        [self._fixed_vars, self._variable_vars, new_fs, other_vars]
                    )
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = ast.unparse(node).strip()
            f_name = ast.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = ast.unparse(node).strip()
            f_name = ast.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k: v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    if '```' in code_str:
        code_str = extract_code(code_str)

    if 'import' in code_str:
        import_pattern = re.compile(r'^\s*(import .*|from .* import .*)$', re.MULTILINE)
        code_str = import_pattern.sub('', code_str).strip()
    assert '__' not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    # print(f'THE CODE STRING IS\n{code_str}\nEND')
    exec(code_str, custom_gvars, lvars)


def extract_code(res):
    if '```python' in res:
        pattern = r'```python\n(.*?)```'
    elif '```Python' in res:
        pattern = r'```Python\n(.*?)```'
    elif '```' in res:
        pattern = r'```\n(.*?)```'
    else:
        pattern = r'.*'
    code_string = re.search(pattern, res, re.DOTALL)
    if not code_string:
        print('input: ', res)
        raise ValueError('extract failed')
    if pattern == r'.*':
        code_string = code_string.group(0).strip()
    else:
        code_string = code_string.group(1).strip()

    lines = code_string.splitlines()
    if '```' in code_string:
        lines = lines[1:]
    lines = [line for line in lines if line.strip() != '']
    code_string = "\n".join(lines)

    return code_string



class LMP_wrapper():

    def __init__(self, env, cfg, render=False):
        self.env = env
        self._cfg = cfg
        self.object_names = self._cfg['env']['init_objs']

        self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
        self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg['env']['coords']['table_z']
        self.render = render

    def is_obj_visible(self, obj_name):

        return obj_name in self.object_names
    
    def reset(self):
        np.random.seed(self.env._seed)
        random.seed(self.env._seed)
        self.env.reset()

    def get_obj_names(self, id=None):
        if not id:
            return self.object_names[::]
        elif isinstance(id, int):
            for s in self.object_names[::]:
                parts = s.split()  # Split the string by spaces
                if parts and parts[-1].isdigit():  # Check if the last part is a number
                    if int(parts[-1]) == id:
                        return [s]  # Return the matching string        
            raise ValueError(f'no matching obj with id {id}')
        elif isinstance(id, (str, np.str_)):
            return [id]
        elif isinstance(id, (list, tuple, np.ndarray)):
            return list(id)
        else:
            raise ValueError('input should be either int or str')

    def denormalize_xy(self, pos_normalized, size=None):
        pos_normalized = np.array([pos_normalized[1], pos_normalized[0]])
        if not size:
            return pos_normalized * self._range_xy + self._min_xy
        else:
            x = size[0]
            y = size[1]
            min_xy = np.array([-x / 2, -y / 2])
            max_xy = np.array([x / 2, y / 2])
            range_xy = max_xy - min_xy
            return pos_normalized * range_xy + min_xy
        
    def denormalize_bbox(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        left = self.denormalize_xy([x_min,y_max])
        right = self.denormalize_xy([x_max,y_min])
        return [left[0], left[1], right[0], right[1]]

    def get_corner_positions(self):
        unit_square = shapely.geometry.box(0, 0, 1, 1)
        normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
        corners = np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))
        return corners

    def get_side_positions(self):
        side_xs = np.array([0, 0.5, 0.5, 1])
        side_ys = np.array([0.5, 0, 1, 0.5])
        normalized_side_positions = np.c_[side_xs, side_ys]
        side_positions = np.array(
            ([self.denormalize_xy(corner) for corner in normalized_side_positions]))
        return side_positions

    def get_obj_pos(self, obj_name, count=1):
        # return the xy position of the object in robot base frame. YHH: Why only xy position?
        return self.env.get_obj_pos(obj_name, count)  # [:2]

    def get_obj_rot(self, obj_name, count=1):

        return self.env.get_obj_rot(obj_name, count)

    def get_obj_positions_np(self, objects):
        if all(type(obj) == int for obj in objects) or all(
                'with obj_id' in obj for obj in objects):
            return [self.get_obj_pos(object)[0] for object in objects]

        positions_dict = {}

        # Retrieve positions for each object type
        for obj in objects:
            if obj not in positions_dict:
                positions_dict[obj] = self.get_obj_pos(obj, -1)

        # Initialize index counters for each object type
        counters = {obj: 0 for obj in positions_dict}

        # List to store the final positions in the required order
        position_list = []

        # Populate the position_list with the correct positions
        for obj in objects:
            pos_index = counters[obj]  # Get the current index for this object type
            position_list.append(positions_dict[obj][pos_index])  # Add the position to the list
            counters[obj] += 1  # Increment the index for this object type

        return position_list

    def get_obj_rotations_np(self, objects):
        if all(type(obj) == int for obj in objects) or all(
                'with obj_id' in obj for obj in objects):
            return [self.get_obj_rot(object)[0] for object in objects]

        rotations_dict = {}

        # Retrieve positions for each object type
        for obj in objects:
            if obj not in rotations_dict:
                rotations_dict[obj] = self.get_obj_rot(obj, -1)

        # Initialize index counters for each object type
        counters = {obj: 0 for obj in rotations_dict}

        # List to store the final positions in the required order
        rotation_list = []

        # Populate the position_list with the correct positions
        for obj in objects:
            pos_index = counters[obj]  # Get the current index for this object type
            rotation_list.append(rotations_dict[obj][pos_index])  # Add the position to the list
            counters[obj] += 1  # Increment the index for this object type

        return rotation_list

    def get_obj_pos_dict(self):
        catalog = {}
        for obj in self.get_obj_names():
            catalog[obj] = [self.get_obj_pos(obj)[0],
                            utils.quatXYZW_to_eulerXYZ(self.get_obj_rot(obj)[0])]
        return catalog

    def get_bbox(self, obj_name):
        # return the axis-aligned object bounding box in robot base frame (not in pixels)
        # the format is (min_x, min_y, max_x, max_y)
        is_zone = False
        is_pallet = False
        scale = 1
        if isinstance(obj_name, (list, np.ndarray, tuple)):
            obj_name = obj_name[0]
        if 'scaled' in self.get_obj_names(obj_name)[0]:
            pattern = r'(\d+(\.\d+)?)x'
            match = re.search(pattern, obj_name)
            scale = float(match.group(1)) if match else 1
        if 'zone' in self.get_obj_names(obj_name)[0]:
            is_zone = True
        if 'pallet' in self.get_obj_names(obj_name)[0]:
            is_pallet = True
        bbox = self.env.get_bounding_box(obj_name)
        size_x = bbox[3] - bbox[0]
        size_y = bbox[4] - bbox[1]
        size_z = bbox[5] - bbox[2]
        size = 50 * scale * np.array([size_x, size_y, size_z]) if is_zone else scale * np.array(
            [size_x, size_y, size_z])
        if is_pallet:
            size *= 0.5
        return tuple(size)

    def get_two_bbox(self, obj_name):
        if isinstance(obj_name, (list, np.ndarray, tuple)):
            obj_name = obj_name[0]
        bbox = self.env.get_bounding_box(obj_name)
        return bbox

    def get_color(self, obj_name):
        for color, rgb in utils.COLORS.items():
            if color in obj_name:
                return rgb

    def pick_place(self, obj, place):
        pick_pos = self.get_obj_pos(obj)[0]
        pick_rot = self.get_obj_rot(obj)[0]
        self.env.step(action={'pose0': (pick_pos, pick_rot), 'pose1': place})

    def put_first_on_second(self, arg1, arg2):
        # put the object with obj_name on top of target
        # target can either be another object name, or it can be an x-y position in robot base frame
        if not arg1 or not arg2:
            print('missing argument')
            return
        if isinstance(arg1, (str, int, np.str_)):
            pick_pos = (self.get_obj_pos(arg1)[0], self.get_obj_rot(arg1)[0])
        else:
            pick = list(arg1)
            if not isinstance(pick[0], (list, tuple, np.ndarray)):
                if len(pick) == 2:
                    pick_pos = ((pick[0], pick[1], 0), (0, 0, 0, 1))
                elif len(pick) == 3:
                    pick_pos = ((pick[0], pick[1], pick[2]), (0, 0, 0, 1))
            else:
                pick_pos = arg1

        if isinstance(arg2, (str, int, np.str_)):
            place_pos = (self.get_obj_pos(arg2)[0], self.get_obj_rot(arg2)[0])
        else:
            place = list(arg2)
            if not isinstance(place[0], (list, tuple, np.ndarray)):
                if len(place) == 2:
                    place_pos = ((place[0], place[1], 0), (0, 0, 0, 1))
                elif len(place) == 3:
                    place_pos = ((place[0], place[1], place[2]), (0, 0, 0, 1))
            else:
                place_pos = arg2
        self.env.step(action={'pose0': pick_pos, 'pose1': place_pos})

    def stack_objects_in_order(self, object_names, targ=None):
        if not object_names:
            return
        if not targ:
            if not isinstance(object_names, (list, tuple, np.ndarray)):
                object_names = [object_names]
            for i in range(len(object_names) - 1):
                self.put_first_on_second(object_names[i + 1], object_names[i])
        else:
            if not isinstance(object_names, (list, tuple, np.ndarray)):
                object_names = [object_names]
            self.put_first_on_second(object_names[0], targ)
            for i in range(len(object_names) - 1):
                self.put_first_on_second(object_names[i + 1], object_names[i])

    def is_target_occupied(self, targ, r=0.02):

        search_range = self.get_obj_names()
        if isinstance(targ, (list, tuple, np.ndarray)):
            if isinstance(targ[0], (list, tuple, np.ndarray)):
                targ = targ[0]
            targ = np.array(targ)[:2]  # force to check in 2d
            dim = len(targ)
            if dim not in [2, 3]:
                raise ValueError("Target position must be either 2D or 3D.")
        elif isinstance(targ, (int, str, np.str_)):
            targ_obj = self.get_obj_names(targ)[0]
            if targ_obj in search_range:
                search_range.remove(targ_obj)
            x, y, _ = self.get_bbox(targ)
            r = 0.5 * np.linalg.norm([x, y])
            targ = self.get_obj_pos(targ)[0]
            targ = np.array(targ)[:2]  # force to check in 2d
            dim = len(targ)
        else:
            raise ValueError("Target must be only one position, id, or name.")

        # Define the center of the circle/sphere
        center = np.array(targ)
        occupied_objects = []

        for obj in search_range:
            # Get the bounding box of the object
            bbox = self.get_two_bbox(obj)

            # Extract bounding box coordinates
            if dim == 2:
                x_min, y_min, _, x_max, y_max, _ = bbox
                closest_point = np.array([
                    np.clip(center[0], x_min, x_max),
                    np.clip(center[1], y_min, y_max)
                ])
            elif dim == 3:
                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                closest_point = np.array([
                    np.clip(center[0], x_min, x_max),
                    np.clip(center[1], y_min, y_max),
                    np.clip(center[2], z_min, z_max)
                ])

            # Calculate the distance from the center to the closest point on the bounding box
            distance_to_bbox = distance.euclidean(center, closest_point)

            # Check if this distance is less than or equal to the radius
            if distance_to_bbox <= r:
                occupied_objects.append(obj)

        return occupied_objects

    def get_random_free_pos(self, targ=None, r=0.02, search_area=None, grid_size=0.002):
        # local function to convert item of listed-targ 
        def convert_to_3d_pos(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                # Handling pos
                if len(item) == 2 and all(isinstance(coord, (int, float)) for coord in item):
                    return [item[0], item[1], 0]
                elif len(item) == 3 and all(isinstance(coord, (int, float)) for coord in item):
                    return list(item)
                # Handling pos_rot
                elif len(item) == 2 and all(
                        isinstance(coord, (list, tuple, np.ndarray)) for coord in item) and len(
                    item[1]) == 4:
                    pos = item[0]
                    if len(pos) == 2:
                        return [pos[0], pos[1], 0]
                    elif len(pos) == 3:
                        return list(pos)
                    else:
                        raise ValueError("pos in pos_rot must be 2D or 3D")
            elif isinstance(item, (str, int, np.str_)):
                # Use get_obj_pos for str or int
                x, y, _ = self.get_bbox(item)
                radius = 0.5 * np.linalg.norm([x, y])
                return (self.get_obj_pos(item)[0], radius)
            else:
                raise ValueError("Unsupported type for conversion to 3D position")

        def convert_targ_to_3d_pos_list(targ):
            # Check if targ is a list that is not a nested list (indicating a single pos)
            if isinstance(targ, (list, tuple, np.ndarray)):
                if all(isinstance(coord, float) for coord in targ) and (
                        len(targ) == 2 or len(targ) == 3):
                    # It's a single position
                    return [convert_to_3d_pos(targ)]
                elif all(isinstance(coord, (list, tuple, np.ndarray)) for coord in targ) and len(
                        targ) == 2 and len(targ[1]) == 4:
                    # It's a single pos_rot
                    return [convert_to_3d_pos(targ)]
                elif all(isinstance(coord, float) for coord in targ) and len(targ) == 4:
                    raise ValueError("Search area")
                else:
                    # It's a list of elements
                    pos_list = [convert_to_3d_pos(item) for item in targ]
                    return pos_list
            else:
                # targ is a single item (str, int)
                return [convert_to_3d_pos(targ)]

        if search_area:
            x_min, y_min, x_max, y_max = search_area
        else:
            x_min, y_min = self._cfg['env']['coords']['top_left']
            x_max, y_max = self._cfg['env']['coords']['bottom_right']

        if targ:
            try:
                targ = convert_targ_to_3d_pos_list(targ)
            except ValueError as e:
                if str(e) == "Search area":
                    x_min, y_min, x_max, y_max = targ
                    targ = []
                else:
                    raise
        else:
            targ = []

        # Generate a grid of potential positions
        x_coords = np.arange(x_min, x_max, grid_size)
        y_coords = np.arange(y_min, y_max, grid_size)
        potential_positions = [(x, y, 0.001) for x in x_coords for y in y_coords]

        # Remove positions that are too close to the target
        for sub_targ in targ:
            if isinstance(sub_targ, tuple):
                potential_positions = [
                    pos for pos in potential_positions
                    if np.linalg.norm(np.array(pos) - np.array(sub_targ[0])) > sub_targ[1]
                ]
            else:
                potential_positions = [
                    pos for pos in potential_positions
                    if np.linalg.norm(np.array(pos) - np.array(sub_targ)) > r
                ]

        # Remove positions that are occupied by objects
        free_positions = [
            pos for pos in potential_positions
            if not self.is_target_occupied(pos, r)
        ]

        if not free_positions:
            print('no suitable position')
            return None  # No free position found

        # Randomly select a free position
        pos = random.choice(free_positions)
        return [list(pos), [0, 0, 0, 1]]

    def get_robot_pos(self):
        # return robot end-effector xy position in robot base frame
        return self.env.get_ee_pos()

    def goto_pos(self, position):
        # move the robot end-effector to the desired xy position while maintaining same z
        ee_xyz = self.env.get_ee_pos()
        if len(position) == 2:
            if not isinstance(position[0], (tuple, list, np.ndarray)):
                position_xyz = [np.concatenate([position, ee_xyz[-1]]), [0, 0, 0, 1]]
        elif len(position) == 3:
            position_xyz = [list(position), [0, 0, 0, 1]]
        else:
            position_xyz = position
        while np.linalg.norm(position_xyz - ee_xyz) > 0.01:
            self.env.movep(position_xyz)
            self.env.step_simulation()
            ee_xyz = self.env.get_ee_pos()

    def follow_traj(self, traj):
        for pos in traj:
            self.goto_pos(pos)

    def get_corner_positions(self):
        # TODO: repetitive name.
        normalized_corners = np.array([
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0]
        ])
        return np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))

    def get_side_positions(self):
        # TODO: repetitive name.
        normalized_sides = np.array([
            [0.5, 1],
            [1, 0.5],
            [0.5, 0],
            [0, 0.5]
        ])
        return np.array(([self.denormalize_xy(side) for side in normalized_sides]))

    def get_corner_name(self, pos):
        corner_positions = self.get_corner_positions()
        if len(pos) == 2:
            corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        elif len(pos) == 3:
            pos = pos[:2]
            corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        return ['top left corner', 'top right corner', 'bottom left corner', 'botom right corner'][
            corner_idx]

    def get_side_name(self, pos):
        side_positions = self.get_side_positions()
        side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
        return ['top side', 'right side', 'bottom side', 'left side'][side_idx]

def set_llm_model(llm_model_name):
    """ globally set llm-model"""
    global model
    model = llm_model_name


def set_max_token(token):
    global max_token
    max_token = token


def update_model():
    global model
    return model


def update_max_token():
    global max_token
    return max_token


def update_memory():
    global mem
    memory = {}
    if not mem[0]:
        for id in mem[1]:
            memory[id] = f'{mem[2]}GB'
    else:
        for id, memo in mem[1]:
            memory[id] = f'{memo}GB'
    return memory


def say(msg):
    global answer
    msg = f'robot says: {msg}'
    answer += (msg + '\n')
    print(msg)


def check_obj():
    object_ids = []
    num_bodies = p.getNumBodies()
    for i in range(num_bodies):
        object_id = p.getBodyUniqueId(i)
        object_ids.append(object_id)
    return object_ids


def setup_LMP(env, cfg_tabletop, llm='gpt4',
              offline_model=None, offline_tokenizer=None, use_vllm=False):
    # LMP env wrapper
    cfg_tabletop = copy.deepcopy(cfg_tabletop)
    cfg_tabletop['env'] = dict()
    cfg_tabletop['env']['init_objs'] = env.object_list
    cfg_tabletop['env']['coords'] = utils.lmp_tabletop_coords
    cfg_tabletop['llm'] = llm
    LMP_env = LMP_wrapper(env, cfg_tabletop)
    # creating APIs that the LMPs can interact with
    fixed_vars = {
        'np': np,
        'utils': utils,
        'itertools': itertools
    }
    fixed_vars.update({
        name: getattr(shapely.geometry, name)
        for name in shapely.geometry.__all__
    })

    fixed_vars.update({
        name: getattr(shapely.affinity, name)
        for name in shapely.affinity.__all__
    })
    variable_vars = {
        k: getattr(LMP_env, k)
        for k in [
            'get_bbox', 'get_obj_pos', 'get_color', 'is_obj_visible', 'denormalize_xy',
            'put_first_on_second', 'get_obj_names', 'get_obj_rot', 'get_obj_positions_np',
            'get_corner_name', 'get_side_name', 'get_obj_rotations_np', 'goto_pos',
            'is_target_occupied', 'get_random_free_pos', 'stack_objects_in_order',
            'get_obj_pos_dict', 'denormalize_bbox', 'reset',
        ]
    }
    variable_vars['say'] = say

    # creating the function-generating LMP
    lmp_fgen = LMPFGen(
        (cfg_tabletop['lmps']['fgen'], cfg_tabletop['llm']),
        fixed_vars,
        variable_vars,
    )

    # creating other low-level LMPs
    variable_vars.update({
        k: LMP(
            k,
            (cfg_tabletop['lmps'][k], cfg_tabletop['llm']),
            lmp_fgen,
            fixed_vars,
            variable_vars,
        )
        for k in [
            'parse_obj_name', 'parse_position',
            'parse_question', 'transform_shape_pts', 'parse_completion'
        ]
    })

    # creating the LMP that deals w/ high-level language commands
    lmp_tabletop_ui = LMP(
        'tabletop_ui',
        (cfg_tabletop['lmps']['tabletop_ui'], cfg_tabletop['llm']),
        lmp_fgen,
        fixed_vars,
        variable_vars,
    )

    return lmp_tabletop_ui


def main(cfg):
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )

    cfg['task'] = cfg['task'].replace("_", "-")
    mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']
    check = cfg['check']
    note = (" Write code to complete the task")

    llm = 'gpt4'
    llm_model_name = "gpt-4o-mini"
    set_llm_model(llm_model_name)

    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path} if save_data")
    print(f"Mode: {mode}")

    seed = dataset.max_seed
    max_eps = 1 * cfg['n']
    if seed < 0:
        seed = -1

    if 'regenerate_data' in cfg:
        dataset.n_episodes = 0

    curr_run_eps = 0

    while dataset.n_episodes < cfg['n'] and curr_run_eps < max_eps:
        answer = ''
        
        # for epi_idx in range(cfg['n']):
        episode = []
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)
        task = tasks.names[cfg['task']]()

        env.seed(seed)
        print('CAP run: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))
        try:
            curr_run_eps += 1  # make sure exits the loop
            # env.seed(seed)
            env.set_task(task)
            env.reset()
            # assert len(env.object_list) == len(check_obj()) - 5

            # Start video recording
            if record:
                env.start_rec(
                    f'{dataset.n_episodes + 1:06d}_CAP') if not check else env.start_rec(
                    f'{dataset.n_episodes + 1:06d}_CAP_check')

            # Rollout LLM policy
            goal = task.goal + ' Finally check the completion of task.' if check else task.goal
            goal = goal + note

            lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, llm)
            lmp_tabletop_ui(goal, f'objects = {env.object_list}')

            if record:
                env.end_rec()

            obs = env._get_obs()
            info = env.info

            file_path = (
                os.path.join(
                    data_path,
                    'answers',
                    f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}.txt'
                )
                if not check
                else os.path.join(
                    data_path,
                    'answers',
                    f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}_check.txt'
                )
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(answer)
            episode.append((obs, None, 1, info))

        except:
            err = str(traceback.format_exc())
            answer += (err + '\n')
            to_print = highlight(f"{err}", PythonLexer(), TerminalFormatter())
            print(to_print)
            if record:
                env.end_rec()

            obs = env._get_obs()
            info = env.info

            file_path = (
                os.path.join(
                    data_path,
                    'answers',
                    f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}.txt'
                )
                if not check
                else os.path.join(
                    data_path,
                    'answers',
                    f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}_check.txt'
                )
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(answer)
            episode.append((obs, None, 0, info))
            continue

            # Only save completed demonstrations.
        if save_data:
            dataset.add(seed, episode)

        if hasattr(env, 'blender_recorder'):
            print("blender pickle saved to ",
                    '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))
            env.blender_recorder.save(
                '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))


cfg_tabletop = {
    'lmps': {
        'tabletop_ui': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_tabletop_ui.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'objects = ['],
            'maintain_session': True,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'whole_answer',
        },
        'parse_obj_name': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_parse_obj_name.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'objects = ['],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'ret_val',
        },
        'parse_position': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_parse_position.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'ret_val',
        },
        'parse_question': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_parse_question.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'objects = ['],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'ret_val',
        },
        'transform_shape_pts': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_transform_shape_pts.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'new_shape_pts',
        },
        'fgen': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_fgen.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# define function: ',
            'query_suffix': '.',
            'stop': ['# define', '# example'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
        },
        'parse_completion': {
            'prompt_text': open(f"ravens/prompts/capravens/prompt_parse_completion.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'object_positions = {'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'judge',
        },
        'VLM': {
            'prompt_text': '',
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0.7,
            'query_prefix': '# ',
            'query_suffix': '.',
            'maintain_session': True,
            'debug_mode': False,
            'has_return': True,
            'return_val_name': 'judge',
        },
    }
}

config = {
    "root_dir": ".",
    "tag": "default",
    "debug": False,
    "gpt_temperature": 0.8,
    "prompt_folder": "vanilla_task_generation_prompt",
    "max_env_run_cnt": 3,
    "trials": 1,
    "output_folder": "output/output_stats/",
    "model_output_dir": "",
    "gpt_model": "gpt-4o",
    "llama_model_name": "",
    "use_vllm": False,
    "n_gpus": 1,
    "openai_key": "",
    "genai_key": "",
    "llama_key": "",
    "hf_token": "",
    "task_description_candidate_num": -1,
    "task_asset_candidate_num": -1,
    "task_code_candidate_num": 4,
    "prompt_data_path": "prompts/data/",
    "save_data": True,
    "save_memory": False,
    "load_memory": False,
    "use_template": False,
    "reflection_agreement_num": 2,
    "target_task_name": "",
    "target_task_description": "",
    "save_code_early": False,
    "load_task_num": -1,

    "data_dir": "ravens/data",
    "assets_root": "/Users/maxfest/vscode/thesis/ravens/environments/assets",
    "disp": True,
    "shared_memory": False,
    "task": "build-house",
    "mode": "cap",
    "n": 1,
    "random": True,
    "save_data": False,
    "check": False,
    "check_using_VLM": False,
    "manual_eval": False,
    "dataset": {
        "type": "single",
        "images": True,
        "cache": True,
        "augment": {
            "theta_sigma": 60
        }
    },
    "record": {
        "save_video": False,
        "save_video_path": "${data_dir}/${task}-cap/videos/",
        "add_text": True,
        "add_task_text": True,
        "fps": 20,
        "video_height": 640,
        "video_width": 720
    }
}



if __name__ == '__main__':
    main(config)

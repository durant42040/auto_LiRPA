#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from torch.nn import Module
from .base import *
from .constant import BoundConstant
from .solver_utils import grb


class BoundAdd(Bound):
    """``onnx::Add`` and raw-JIT ``aten::add`` (optional ``alpha`` scalar input)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        options = options or {}
        # FIXME: This is not the right way to enable patches mode.
        # Instead we must traverse the graph and determine when patches mode needs to be used.

        self.mode = options.get("conv_mode", "matrix")
        # JIT ``aten::add`` carries a third ``alpha`` operand (``out = x + alpha*y``);
        # ONNX ``Add`` has only two.  Detect the JIT layout so we route the extra
        # scalar slot through ``bound_backward``/``interval_propagate`` cleanly.
        self._jit_layout = (attr or {}).get('_graph_op') == 'aten::add'

    @staticmethod
    def _coerce_alpha(alpha):
        if alpha is None:
            return 1
        if isinstance(alpha, torch.Tensor):
            return alpha.item() if alpha.numel() == 1 else alpha
        return alpha

    def forward(self, x, y, alpha=1):
        self.x_shape = x.shape
        self.y_shape = y.shape
        alpha = self._coerce_alpha(alpha)
        if alpha == 1:
            return x + y
        return x + y * alpha

    def bound_backward(self, last_lA, last_uA, x, y, *extras, **kwargs):
        if extras:
            # ``extras[0]`` is the alpha-slot ``Bound`` node (e.g. a
            # ``BoundConstant`` holding ``1`` for the JIT ``aten::add(x, y,
            # alpha)`` form).  Read its scalar value, not the node itself.
            alpha_node = extras[0]
            alpha_val = getattr(
                alpha_node, 'value', getattr(alpha_node, 'forward_value', alpha_node)
            )
            alpha = self._coerce_alpha(alpha_val)
            if alpha != 1:
                raise NotImplementedError(
                    "BoundAdd CROWN with alpha != 1 is not supported."
                )

        def _bound_oneside(last_A, w):
            if last_A is None:
                return None
            return self.broadcast_backward(last_A, w)

        uA_x = _bound_oneside(last_uA, x)
        uA_y = _bound_oneside(last_uA, y)
        lA_x = _bound_oneside(last_lA, x)
        lA_y = _bound_oneside(last_lA, y)
        result = [(lA_x, uA_x), (lA_y, uA_y)]
        if extras:
            result.append((None, None))
        return result, 0, 0

    def bound_forward(self, dim_in, x, y, *extras):
        if extras:
            alpha = self._coerce_alpha(getattr(extras[0], 'value', extras[0]))
            if alpha != 1:
                raise NotImplementedError(
                    "BoundAdd bound_forward with alpha != 1 is not supported."
                )
        lb, ub = x.lb + y.lb, x.ub + y.ub

        def add_w(x_w, y_w, x_b, y_b):
            if x_w is None and y_w is None:
                return None
            elif x_w is not None and y_w is not None:
                return x_w + y_w
            elif y_w is None:
                return x_w + torch.zeros_like(y_b)
            else:
                return y_w + torch.zeros_like(x_b)

        lw = add_w(x.lw, y.lw, x.lb, y.lb)
        uw = add_w(x.uw, y.uw, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y, *extras):
        if extras:
            alpha_l = self._coerce_alpha(extras[0][0])
            alpha_u = self._coerce_alpha(extras[0][1])
            if alpha_l != 1 or alpha_u != 1:
                raise NotImplementedError(
                    "BoundAdd IBP with alpha != 1 is not supported."
                )
        assert (not isinstance(y, Tensor))
        return x[0] + y[0], x[1] + y[1]

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            # constants if both inputs are tensors
            self.solver_vars = self.forward(v[0], v[1])
            return
        # we have both gurobi vars as inputs
        this_layer_shape = self.output_shape
        gvar_array1 = np.array(v[0])
        if isinstance(v[1], Tensor):
            var2 = v[1].cpu().numpy()
            # flatten to create vars and constrs first
            gvar_array1 = gvar_array1.reshape(-1)
            new_layer_gurobi_vars = []
            for neuron_idx, var1 in enumerate(gvar_array1):
                var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                                   vtype=grb.GRB.CONTINUOUS,
                                   name=f'lay{self.name}_{neuron_idx}')
                model.addConstr(var == (var1 + var2), name=f'lay{self.name}_{neuron_idx}_eq')
                new_layer_gurobi_vars.append(var)
        else:
            gvar_array2 = np.array(v[1])
            assert gvar_array1.shape == gvar_array2.shape and gvar_array1.shape == this_layer_shape[1:]

            # flatten to create vars and constrs first
            gvar_array1 = gvar_array1.reshape(-1)
            gvar_array2 = gvar_array2.reshape(-1)
            new_layer_gurobi_vars = []
            for neuron_idx, (var1, var2) in enumerate(zip(gvar_array1, gvar_array2)):
                var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'lay{self.name}_{neuron_idx}')
                model.addConstr(var == (var1 + var2), name=f'lay{self.name}_{neuron_idx}_eq')
                new_layer_gurobi_vars.append(var)
        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()

    def build_gradient_node(self, grad_upstream):
        if not self.inputs[0].no_jacobian:
            grad0_node = AddGrad(self.inputs[0].output_shape if self.inputs[0].batch_dim != -1 else
                                 torch.Size((1,) + self.inputs[0].output_shape))
            grad0 = (grad0_node, (grad_upstream,), [])
        else:
            grad0 = None
        if not self.inputs[1].no_jacobian:
            grad1_node = AddGrad(self.inputs[1].output_shape if self.inputs[1].batch_dim != -1 else
                                 torch.Size((1,) + self.inputs[1].output_shape))
            grad1 = (grad1_node, (grad_upstream,), [])
        else:
            grad1 = None
        return [grad0, grad1]


class BoundSub(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # FIXME: This is not the right way to enable patches mode. Instead we must traverse the graph and determine when patches mode needs to be used.
        self.mode = options.get("conv_mode", "matrix")

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def bound_backward(self, last_lA, last_uA, x, y, **kwargs):
        def _bound_oneside(last_A, w, sign=-1):
            if last_A is None:
                return None
            if isinstance(last_A, torch.Tensor):
                return self.broadcast_backward(sign * last_A, w)
            elif isinstance(last_A, Patches):
                if sign == 1:
                    # Patches shape requires no broadcast.
                    return last_A
                else:
                    # Multiply by the sign.
                    return last_A.create_similar(sign * last_A.patches)
            else:
                raise ValueError(f'Unknown last_A type {type(last_A)}')

        uA_x = _bound_oneside(last_uA, x, sign=1)
        uA_y = _bound_oneside(last_uA, y, sign=-1)
        lA_x = _bound_oneside(last_lA, x, sign=1)
        lA_y = _bound_oneside(last_lA, y, sign=-1)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        lb, ub = x.lb - y.ub, x.ub - y.lb

        def add_w(x_w, y_w, x_b, y_b):
            if x_w is None and y_w is None:
                return None
            elif x_w is not None and y_w is not None:
                return x_w + y_w
            elif y_w is None:
                return x_w + torch.zeros_like(y_b)
            else:
                return y_w + torch.zeros_like(x_b)

        # Some nodes such as BoundConstant does not have uw and lw.
        lw = add_w(x.lw, -y.uw if y.uw is not None else None, x.lb, y.lb)
        uw = add_w(x.uw, -y.lw if y.lw is not None else None, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        return x[0] - y[1], x[1] - y[0]

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            # constants if both inputs are tensors
            self.solver_vars = self.forward(v[0], v[1])
            return
        # we have both gurobi vars as inputs
        this_layer_shape = self.output_shape
        gvar_array1 = np.array(v[0])
        gvar_array2 = np.array(v[1])
        assert gvar_array1.shape == gvar_array2.shape and gvar_array1.shape == this_layer_shape[1:]

        # flatten to create vars and constrs first
        gvar_array1 = gvar_array1.reshape(-1)
        gvar_array2 = gvar_array2.reshape(-1)
        new_layer_gurobi_vars = []
        for neuron_idx, (var1, var2) in enumerate(zip(gvar_array1, gvar_array2)):
            var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f'lay{self.name}_{neuron_idx}')
            model.addConstr(var == (var1 - var2), name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()

    def build_gradient_node(self, grad_upstream):
        if not self.inputs[0].no_jacobian:
            grad_node_0 = AddGrad(self.inputs[0].output_shape if self.inputs[0].batch_dim != -1 else
                                  torch.Size((1,) + self.inputs[0].output_shape), w=1.0)
            grad0 = (grad_node_0, (grad_upstream,), [])
        else:
            grad0 = None
        if not self.inputs[1].no_jacobian:
            grad_node_1 = AddGrad(self.inputs[1].output_shape if self.inputs[1].batch_dim != -1 else
                                  torch.Size((1,) + self.inputs[1].output_shape), w=-1.0)
            grad1 = (grad_node_1, (grad_upstream,), [])
        else:
            grad1 = None
        return [grad0, grad1]


class AddGrad(Module):
    def __init__(self, input_shape, w=1.0):
        super().__init__()
        # We need the input shape to handle broadcasting.
        self.input_shape = input_shape
        self.w = w

    def forward(self, grad_last):
        return reduce_broadcast_dims(grad_last * self.w, self.input_shape)

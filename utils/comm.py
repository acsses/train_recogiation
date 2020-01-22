# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class CommunicatorWrapper(object):
    def __init__(self, ctx):
        try:
            import nnabla.communicators as C
            comm = C.MultiProcessDataParallelCommunicator(ctx)
        except Exception as e:
            print(e)
            print('No communicator found. Running with a single process. If you run this with MPI processes,'
                  ' all processes will perform totally same.')
            self.n_procs = 1
            self.rank = 0 if ctx.context == "cpu" else int(ctx.device_id)
            self.ctx = ctx
            self.comm = None
            return

        comm.init()
        self.n_procs = comm.size
        self.rank = comm.rank
        self.ctx = ctx
        self.ctx.device_id = str(self.rank)
        self.comm = comm

    def all_reduce(self, params, division, inplace):
        if self.n_procs == 1:
            # skip all reduce since no processes have to be all-reduced
            return
        self.comm.all_reduce(params, division=division, inplace=inplace)

    def all_reduced_solver_update(self, solver, division=False, inplace=True):
        if self.n_procs > 1:
            params = [
                x.grad for x in solver.get_parameters().values()]
            self.all_reduce(params, division=division, inplace=inplace)

        solver.update()

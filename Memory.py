import numpy as np
import h5py


class Memory:

    def __init__(self, capacity=None):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer[key]

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def clear(self):
        self.buffer.clear()

    def add(self, entry):
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def array(self):
        return np.array(self.buffer)

    def save(self, file, name):
        file.create_dataset(name, data=self.array())

    def load(self, file, name):
        for element in file[name]:
            self.add(element)

    def end_episode(self):
        pass

    @staticmethod
    def shuffled_sample(replays, batch_size, distribution=None):  # uniform if not specified
        """
        :return: List of numpy arrays of a shuffled subset of replays & shuffled indices
        """
        length = len(replays[0])

        assert batch_size <= length
        for replay in replays:
            assert len(replay) == length

        indices = np.random.choice(np.arange(length), size=batch_size, replace=False, p=distribution)
        arrays = [np.empty((batch_size, *replay[0].shape))
                  if isinstance(replay[0], np.ndarray)
                  else np.empty(batch_size)
                  for replay in replays]
        for idx, random_idx in enumerate(indices):
            for array, replay in zip(arrays, replays):
                array[idx] = replay[random_idx]
        return arrays, indices


class ETDMemory(Memory):

    def __init__(self, num_time_steps, void_state, capacity):
        super().__init__(capacity)
        self.num_time_steps = num_time_steps
        self.buffer = [void_state]
        self.ndxs = []
        self.step_ndxs = np.zeros(self.num_time_steps, dtype=np.int)

    def __len__(self):
        return len(self.ndxs)

    def add(self, entry):

        self.step_ndxs = np.roll(self.step_ndxs, -1)
        self.step_ndxs[-1] = len(self.buffer)
        self.ndxs.append(self.step_ndxs)
        self.buffer.append(entry)

    def __getitem__(self, key):
        return self.buffer[key + 1 if key >= 0 else key]

    def __setitem__(self, key, value):
        self.buffer[key + 1 if key >= 0 else key] = value

    def array(self):
        return np.array(self.buffer)[np.array(self.ndxs)]

    def reset(self):

        void_state = self.buffer[0]
        self.buffer.clear()
        self.buffer.append(void_state)
        self.ndxs.clear()
        self.step_ndxs = np.zeros(self.num_time_steps, dtype=np.int)

    def end_episode(self):
        self.step_ndxs = np.zeros(self.num_time_steps, dtype=np.int)

    def save(self, file, name):

        file.create_dataset(f'{name}_buffer', data=np.array(self.buffer))
        file.create_dataset(f'{name}_ndxs', data=np.array(self.ndxs))

    def load(self, file, name):
        for element in file[f'{name}_ndxs']:
            self.ndxs.append(element)
            if element.shape != (self.num_time_steps,):
                raise ValueError('Cannot load dataset: '
                                 'invalid number of time steps')

        for element in file[f'{name}_buffer']:
            self.buffer.append(element)

    @staticmethod
    def shuffled_sample(memories, subset_size, weights=None):
        length = len(memories[0])
        if subset_size > length:
            raise ValueError(f'Subset size {subset_size} is '
                             f'greater than memory length {length}')
        for memory in memories:
            if len(memory) != length:
                raise ValueError('Memories are not all the same length.')
            if not isinstance(memory, Memory):
                raise TypeError('Memories must also be Memory '
                                'or subclass instances')
        indexes = np.random.choice(np.arange(len(memories[0])),
                                   size=subset_size, replace=False,
                                   p=weights)
        arrays = []
        for memory in memories:
            if isinstance(memory, ETDMemory):
                arrays.append(np.empty((subset_size,
                                        memory.num_time_steps,
                                        *memory.buffer[0].shape)))
            elif isinstance(memory[0], np.ndarray):
                arrays.append(np.empty((subset_size, *memory[0].shape)))
            else:
                arrays.append(np.empty(subset_size))
        for ndx, rndx in enumerate(indexes):
            for array, memory in zip(arrays, memories):
                if isinstance(memory, ETDMemory):
                    for andx, sndx in enumerate(memory.ndxs[rndx]):
                        array[ndx, andx] = memory.buffer[sndx]
                else:
                    array[ndx] = memory[rndx]
        return arrays, indexes


class RingMemory(Memory):
    def __init__(self, max_len):
        Memory.__init__(self, capacity=max_len)

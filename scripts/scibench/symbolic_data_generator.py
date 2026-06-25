import numpy as np
import json


def irregular_time_sequence(true_trajectories, time_sequence, time_sequence_drop_rate):
    random_mask = np.random.choice([0, 1],
                                   size=(true_trajectories.shape[0]),
                                   p=[time_sequence_drop_rate, 1 - time_sequence_drop_rate])

    # Apply the mask to the time steps
    masked_true_traj = true_trajectories[random_mask != 0]

    masked_time_sequence = time_sequence[random_mask != 0]
    return masked_true_traj, masked_time_sequence


def compute_time_derivative(batch_of_trajectories, t_evals):
    """
    compute the time derivative through finite difference
    batch_of_trajectories: [batch_size, time_steps, n_vars]
    delta_t: time difference
    return [batch_size, time_steps-1, n_vars]
    """
    shaped = batch_of_trajectories.shape
    time_derivative = np.zeros((shaped[0], shaped[1] - 1, shaped[2]))
    for ti in range(len(t_evals) - 1):
        delta_t = t_evals[ti + 1] - t_evals[ti]
        time_derivative[:, ti, :] = (batch_of_trajectories[:, ti + 1, :] - batch_of_trajectories[:, ti, :]) / delta_t
    return batch_of_trajectories[:, :-1, :], time_derivative


class DataX(object):
    def __init__(self, vars_range_and_types):
        list_of_samplers = json.loads(vars_range_and_types)
        self.data_X_samplers = []
        for one_sample in list_of_samplers:
            if one_sample['name'] == 'Uniform':
                self.data_X_samplers.append(UniformSampling(one_sample['range'], one_sample['only_positive']))
            elif one_sample['name'] == 'LogUniform':
                self.data_X_samplers.append(LogUniformSampling(one_sample['range'], one_sample['only_positive']))
            elif one_sample['name'] == 'IntegerUniform':
                self.data_X_samplers.append(IntegerSampling(one_sample['range'], one_sample['only_positive']))

    def randn(self, sample_size, one_region=None):
        """
        :param sample_size: batch size
        the left and right range of each variable
        :return: return [#input_variables, sample_size, dimension of each variables]
        """
        if one_region is None:
            list_of_X = [one_sampler(sample_size) for one_sampler in self.data_X_samplers]
        else:
            list_of_X = [one_sampler(sample_size, ranged) for one_sampler, ranged in
                         zip(self.data_X_samplers, one_region)]
        return np.stack(list_of_X, axis=0).squeeze()

    def rand_draw_regions(self, num_of_regions, width_fraction=1):
        """
        width_fraction in (0, 1). defined as the fraction of the original variable range.
        """
        list_of_X = [one_sampler(sample_size=num_of_regions) for one_sampler in self.data_X_samplers]
        list_of_X = np.asarray(list_of_X).T
        regions = []

        for j in range(num_of_regions):
            one_region = []
            for i, xi in enumerate(list_of_X[j]):
                width = (self.data_X_samplers[i].range[1] - self.data_X_samplers[i].range[0]) * width_fraction
                one_region.append((xi, min(xi + width, self.data_X_samplers[i].range[1])))
                regions.append(one_region)
        return regions


class DefaultSampling(object):
    def __init__(self, name, range, only_positive=False):
        self.name = name
        self.range = range
        self.only_positive = only_positive

    def draw_regions(self):
        return

    def to_dict(self):
        return {'name': self.name,
                'range': self.range,
                'only_positive': self.only_positive}


class LogUniformSampling(DefaultSampling):
    def __init__(self, ranges, only_positive=False):
        super().__init__('LogUniform', ranges, only_positive)

    def __call__(self, sample_size, ranged=None):
        if ranged is None:
            ranged = self.range
        if self.only_positive:
            # x ~ U(0.1, 10.0)
            log10_min = np.log10(ranged[0])
            log10_max = np.log10(ranged[1])
            return 10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)
        else:
            # x ~ either U(0.0, 1.0) or U(-1.0, 0.) with 50% chance
            num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
            num_negatives = sample_size - num_positives
            log10_min = np.log10(ranged[0])
            log10_max = np.log10(ranged[1])
            pos_samples = 10.0 ** np.random.uniform(log10_min, log10_max, size=num_positives)
            neg_samples = -10.0 ** np.random.uniform(log10_min, log10_max, size=num_negatives)
            all_samples = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(all_samples)
            return all_samples


class UniformSampling(DefaultSampling):
    def __init__(self, ranges, only_positive=False):
        super().__init__('Uniform', ranges, only_positive)

    def __call__(self, sample_size, ranged=None):
        if ranged is None:
            ranged = self.range
        if self.only_positive:
            # x ~ U(0.0, 1.0)
            return np.random.uniform(ranged[0], ranged[1], size=sample_size)
        else:
            num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
            num_negatives = sample_size - num_positives
            pos_samples = np.random.uniform(ranged[0], ranged[1], size=num_positives)
            neg_samples = -np.random.uniform(ranged[0], ranged[1], size=num_negatives)
            all_samples = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(all_samples)
            return all_samples


class IntegerSampling(DefaultSampling):
    def __init__(self, ranges, only_positive=False):
        ranges = [int(ranges[0]), int(ranges[1])]
        super().__init__('IntegerUniform', ranges, only_positive)

    def __call__(self, sample_size, ranged=None):
        if ranged is None:
            ranged = self.range
        if self.only_positive:
            # x ~ U(1, 100)
            return np.random.randint(self.range[0], self.range[1], size=sample_size)
        else:
            # x ~ either U(1, 100) or U(-100, -1) with 50% chance
            num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
            num_negatives = sample_size - num_positives
            pos_samples = np.random.randint(self.range[0], self.range[1], size=num_positives)
            neg_samples = -np.random.randint(self.range[0], self.range[1], size=num_negatives)
            all_samples = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(all_samples)
            return all_samples

    def get_x_grid(self):
        if self.only_positive:
            return np.arange(self.range[0], self.range[1])
        else:
            pos_grid = np.arange(self.range[0], self.range[1])
            neg_grid = -np.arange(self.range[0], self.range[1])
            all_grids = np.concatenate([neg_grid, pos_grid])
            return all_grids

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np

class BuckleyLev():
    
    def __init__(self):
        self.params = {
            "viscosity_o": 1e-3,
            "viscosity_w": 1e-3,
            "initial_sw": 0.2,
            "residual_w": 0.5,
            "residual_o": 0.3,
            "krwe": 1,
            "kroe": 0.99,
            'vd_array': [],
            'poro': 0.24,
            "inject_rate": 20,
            "x-area": 30
        }

    def k_rw(self, sw):
        p = 11.174
        return (self.params['krwe']) * sw**p

    def k_rn(self, sw):
        q = 3.326
        return (1. - self.params['kroe'] * sw)**q

    def fractional_flow(self, sw):
        return 1. / (1. + ((self.k_rn(sw) / self.k_rw(sw)) * (self.params["viscosity_w"] / self.params["viscosity_o"])))

    def fractional_flow_deriv(self, sw):
        h = 0.0001
        return (self.fractional_flow(sw + h) - self.fractional_flow(sw)) / h

    def fractional_flow_2deriv(self, sw):
        h = 0.01
        return ((self.fractional_flow(sw + h) - 2 * self.fractional_flow(sw) + self.fractional_flow(sw - h)) / (h**2))

    def plot_fractional_flow_deriv(self):
        y = torch.linspace(self.params["residual_w"] + 1e-3, 1 - self.params["residual_o"] + 1e-3, 50)
        x = torch.tensor([self.fractional_flow_deriv(i.item()) for i in y])
        
        plt.plot(x.numpy(), y.numpy())
        plt.title('Derivative of fractional flow curve')
        plt.xlabel('dfw/dSw')
        plt.ylabel('Sw')
        plt.show()

    def sw_at_shock_front(self):
        sw_start = self.params['residual_w']
        sw_end = 1. - self.params['residual_o']
        
        for sw in torch.arange(sw_start, sw_end, 0.001):
            if self.fractional_flow_2deriv(sw.item()) < -1e-2:
                sw_start = sw
                break
        
        for sw in torch.arange(sw_end, sw_start, -0.001):
            if self.fractional_flow_2deriv(sw.item()) < -1e-2:
                sw_end = sw
                break
        
        sw_at_front = 0.
        current_min = 1000.
        
        for sw in torch.arange(sw_start, sw_end, 0.0001):
            sw = sw.item()
            current_diff = abs(self.fractional_flow_deriv(sw) - self.fractional_flow(sw) / sw)
            if current_diff < current_min:
                current_min = current_diff
                sw_at_front = sw
        
        return sw_at_front

    def plot_fractional_flow(self):
        x = torch.linspace(self.params["residual_w"] + 1e-3, 1, 100)
        y = torch.tensor([self.fractional_flow(i.item()) for i in x])
        
        plt.plot(x.numpy(), y.numpy())
        plt.title('Fractional flow as a function of water saturation')
        plt.xlabel('Sw')
        plt.ylabel('Fractional flow')
        plt.ylim([0, 1.1])
        plt.xlim([0, 1])
        plt.hlines(y[-1].item(), 0, 1, linestyles='dashed', lw=2, colors='0.4')
        plt.annotate(f'fw max: {y[-1].item():.4f}', xy=(0.08, 0.95))
        plt.show()
        print(y[-1].item())

    def displacement_plot(self):
        v_sh = self.sw_at_shock_front()
        y = torch.linspace(self.params["residual_w"] + 1e-3, 1 - self.params["residual_o"] + 1e-3, 50)
        x = torch.tensor([self.fractional_flow_deriv(i.item()) for i in y if self.fractional_flow_deriv(i.item()) > v_sh])
        return x

    def rarefaction_plot(self):
        x = torch.linspace(self.params["residual_w"] + 1e-3, 1 - self.params["residual_o"] + 1e-3, 50)
        maximum, sw_shock = 0, 0
        grads = []
        
        for swi in x:
            swi = swi.item()
            grad = self.fractional_flow(swi) / (swi - self.params["residual_w"] + 1e-3)
            grads.append(grad)
            if grad > maximum:
                sw_shock = swi
                maximum = grad
        
        rarefaction = sorted(grads[:([i for i, j in enumerate(grads) if j == maximum][0] + 1)], reverse=True)
        y = x[:(len(rarefaction))]
        
        print(rarefaction[0])
        for v in rarefaction:
            self.params['vd_array'].append(v)
        
        plt.plot(rarefaction, y.numpy(), 'b', lw=2)
        plt.plot(rarefaction[0], y[0].item(), 'ro')
        plt.vlines(rarefaction[0], y[0].item(), self.params["initial_sw"], 'b', lw=2)
        plt.hlines(self.params["initial_sw"], rarefaction[0], rarefaction[0] + 1, 'b', lw=2)
        plt.hlines(self.params["initial_sw"], 0, rarefaction[0], linestyles='dashed', lw=2, colors='0.4')
        plt.hlines(y[0].item(), 0, rarefaction[0], linestyles='dashed', lw=2, colors='0.4')
        plt.vlines(rarefaction[0], -2, self.params["initial_sw"], linestyles='dashed', lw=2, colors='0.4')
        
        plt.annotate('V shock', xy=(rarefaction[0] + 0.02, y[0].item() + 0.02))
        plt.annotate('Sw f', xy=(0 + 0.04, y[0].item() - 0.05))
        plt.annotate('Sw i', xy=(0 + 0.04, self.params["initial_sw"] - 0.05))
        
        if rarefaction[-1] > 0.001:
            plt.hlines(y[-1].item(), 0, rarefaction[-1], 'b', lw=2)
            plt.plot(rarefaction[-1], y[-1].item(), 'ro')
            plt.annotate('V min', xy=((rarefaction[-1] + 0.08), (y[-1].item())))
            plt.vlines(rarefaction[-1], y[-1].item(), 0, linestyles='dashed', lw=2, colors='0.4')
            plt.annotate('Sw c', xy=(0 + 0.04, 1 - self.params["residual_o"] + 0.03))

        plt.xlabel('Dimensionless Velocity, Vd = xd/td')
        plt.ylabel('Saturation')
        plt.ylim([0, 1])
        plt.xlim([0, rarefaction[0] + 1])
        plt.title('Saturation profile velocity')
        plt.show()

    def BL_time_sol(self, time):
        sw_at_front = self.sw_at_shock_front()
        sw_deriv_at_front = self.fractional_flow_deriv(sw_at_front)
        print('sw_atfront', sw_at_front)
        print('vel at front', sw_deriv_at_front)
        td = (time * self.params['inject_rate']) / self.params['poro'] * self.params['x-area']
        xd_shock = sw_deriv_at_front * td
        print('xd_shock', xd_shock)
        
        y = torch.arange(1. - self.params["residual_o"], sw_at_front, -0.001)
        y = torch.cat([y, torch.tensor([sw_at_front])])
        
        x = torch.tensor([self.fractional_flow_deriv(sw.item()) for sw in y])
        x2 = x * td
        
        def saturation_at_xd(xd):
            if xd < xd_shock:
                return F.interpolate(y.unsqueeze(0).unsqueeze(0), size=1, mode='linear', align_corners=True)
            else:
                return self.params['initial_sw']
        
        return saturation_at_xd
    
    def generate_dataset(self, num_samples=1000, time_range=(0.1, 10), xd_range=(0, 2)):
        dataset = []
        for _ in range(num_samples):
            # Randomly sample input parameters
            time = np.random.uniform(*time_range)
            xd = np.random.uniform(*xd_range)
            
            # Solve for the saturation profile
            sat_profile = self.BL_time_sol(time)
            
            # Get the saturation at the given xd
            sw = sat_profile(xd)
            
            # Add to dataset
            dataset.append({
                'time': time,
                'xd': xd,
                'sw': sw.item() if isinstance(sw, torch.Tensor) else sw
            })
        
        return dataset


# Usage
bl = BuckleyLev()
dataset = bl.generate_dataset()

# Convert to PyTorch tensors
X = torch.tensor([[d['time'], d['xd']] for d in dataset], dtype=torch.float32)
y = torch.tensor([d['sw'] for d in dataset], dtype=torch.float32)

bl.plot_fractional_flow()
bl.displacement_plot()

print(f"Dataset shape: X {X.shape}, y {y.shape}")
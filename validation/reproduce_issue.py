
def calculate_equilibrium(eff, loss, t_outlet, t_outdoor, q_ext):
    numerator = eff * t_outlet + loss * t_outdoor + q_ext
    denominator = eff + loss
    return numerator / denominator

# Parameters from log
eff = 0.3093
loss = 1.8697
t_outlet = 65.0
t_outdoor = 1.5
q_ext = 0.184  # 184W * 0.001

t_eq = calculate_equilibrium(eff, loss, t_outlet, t_outdoor, q_ext)
print(f"Equilibrium Temperature: {t_eq:.2f}°C")
print(f"Target: 21.2°C")
print(f"Difference: {t_eq - 21.2:.2f}°C")

import numpy as np
import matplotlib.pyplot as plt

def generate_sine(
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        sample_rate: int = 1000,
        duration: float = 1.0
):
    """
    Generate a sampled sine wave.
    
    Parameters:
    -----------
    frequency : float
        Frequency of the sine wave in Hz
    amplitude : float
        Amplitude of the sine wave
    phase : float
        Phase offset in radians
    sample_rate : int
        Number of samples per second in Hz
    duration : float
        Signal duration in seconds
        
    Returns:
    --------
    t : np.ndarray
        Sample times.
    signal : np.ndarray
        Signal values at each sample time
    """

    num_samples = int(sample_rate * duration)

    t = np.arange(num_samples) / sample_rate

    signal = amplitude * np.sin(
        2 * np.pi * frequency * t + phase
    )

    return t, signal

def generate_square(
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        sample_rate: int = 1000,
        duration: float = 1.0
):
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    signal = amplitude * np.sign(
        np.sin(2 * np.pi * frequency * t + phase)
    )

    return t, signal

def generate_triangle(
        frequency: float,
        amplitude: float,
        phase: float = 0.0,
        sample_rate: int = 1000,
        duration: float = 1.0
):
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    signal = (
        (2 * amplitude / np.pi)
        * np.arcsin(
            np.sin(2 * np.pi * frequency * t + phase)
        )
    )

    return t, signal

def generate_multitone(
        frequencies: list[float],
        amplitudes: list[float],
        phases: list[float],
        sample_rate: int = 1000,
        duration: float = 1.0
):
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    signal = np.zeros(num_samples)

    for frequency, amplitude, phase in zip(
        frequencies,
        amplitudes,
        phases
    ):
        signal += amplitude * np.sin(
            2 * np.pi * frequency * t + phase
        )

    return t, signal

def add_noise(
        signal: np.ndarray,
        snr_db: float 
):
    """
    Add zero-mean Gaussian noise to a signal.
    
    Parameters:
    -----------
    signal: np.ndarray
        Clean input signal
    snr_db : float
        Desired signal-to-noise ratio in decibels
        
    Returns:
    --------
    noisy_signal : np.ndarray
        Signal with Guassian noise added.
    """

    signal_power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape
    )

    noisy_signal = signal + noise

    return noisy_signal


if __name__ == "__main__":

    t, signal = generate_multitone(
        frequencies=[10, 30],
        amplitudes=[1.0, 0.5],
        phases=[0.0, 0.0],
        sample_rate=1000,
        duration=1
    )

    noisy_signal = add_noise(
        signal,
        snr_db=30
    )

    plt.plot(t, signal, label="Clean")
    plt.plot(t, noisy_signal, label="Noisy")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Triangle Wave")
    plt.xlim(0, 0.3)

    plt.legend()
    plt.grid(True)
    plt.show()
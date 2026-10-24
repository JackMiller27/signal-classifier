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
    t, clean_signal = generate_sine(
        frequency=10,
        amplitude=2,
        phase=0,
        sample_rate=1000,
        duration=1
    )

    noisy_signal = add_noise(
        clean_signal,
        snr_db=30
    )

    plt.plot(t, clean_signal, label="Clean")
    plt.plot(t, noisy_signal, label="Noisy")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Clean vs. Noisy 10 Hz Sine Wave")
    plt.xlim(0, 0.3)

    plt.legend()
    plt.grid(True)
    plt.show()
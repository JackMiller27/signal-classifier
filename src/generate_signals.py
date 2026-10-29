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
    """
    Generate a sampled square wave.
    
    Parameters:
    -----------
    frequency : float
        Frequency of the square wave in Hz
    amplitude : float
        Amplitude of the square wave
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
    """
    Generate a sampled triangle wave.
    
    Parameters:
    -----------
    frequency : float
        Frequency of the triangle wave in Hz
    amplitude : float
        Amplitude of the triangle wave
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
    """
    Generate a sampled multitone wave.
    
    Parameters:
    -----------
    frequency : list[float]
        Frequencies of the sine waves in Hz
    amplitude : list[float]
        Amplitudes of the sine waves
    phase : list[float]
        Phase offsets in radians
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

def generate_random_signal(
    sample_rate: int = 1000,
    duration: float = 1.0,
    min_frequency: float = 5.0,
    max_frequency: float = 100.0,
    min_amplitude: float = 0.5,
    max_amplitude: float = 2.0,
    min_snr: float = -5.0,
    max_snr: float = 30.0
):
    """
        Generate a random sampled wave from the types: sine, square, triangle, multitone.
        
        Parameters:
        -----------
        phase : float
            Phase offset in radians
        sample_rate : int
            Number of samples per second in Hz
        duration : float
            Signal duration in seconds
        min_frequency: float
            Minimum possible value for frequency in Hz
        max_frequency: float
            Maximum possible value for frequency in Hz
        min_amplitude: float
            Minimum possible value for amplitude
        max_amplitude: float
            Maximum possible value for amplitude
        min_phase: float
            Minimum possible value for phase
        max_phase: float
            Maximum possible value for phase
        min_snr: float
            Minimum possible value of SNR
        max_snr: float
            Maximum possible vlaue of SNR
            
        Returns:
        --------
        t : np.ndarray
            Sample times.
        signal : np.ndarray
            Signal values at each sample time
        """
    nyquist_frequency = sample_rate / 2

    if max_frequency >= nyquist_frequency:
        raise ValueError(
            "max_frequency must be below the Nyquist frequency "
            f"({nyquist_frequency} Hz)"
        )

    
    signal_type = np.random.choice(
        ["sine", "square", "triangle", "multitone"]
    )

    frequency = np.random.uniform(
        min_frequency,
        max_frequency
    )

    amplitude = np.random.uniform(
        min_amplitude,
        max_amplitude
    )

    phase = np.random.uniform(
        0,
        2 * np.pi
    )

    snr_db = np.random.uniform(
        min_snr,
        max_snr
    )

    if signal_type == "sine":
        t, clean_signal = generate_sine(
            frequency,
            amplitude,
            phase,
            sample_rate,
            duration
        )

    elif signal_type == "square":
        t, clean_signal = generate_square(
            frequency,
            amplitude,
            phase,
            sample_rate,
            duration
        )

    elif signal_type == "triangle":
        t, clean_signal = generate_triangle(
            frequency,
            amplitude,
            phase,
            sample_rate,
            duration
        )

    else:
        second_frequency = np.random.uniform(
            min_frequency,
            max_frequency
        )

        second_amplitude = np.random.uniform(
            min_amplitude,
            max_amplitude
        )

        second_phase = np.random.uniform(
            0,
            2 * np.pi
        )

        t, clean_signal = generate_multitone(
            frequencies=[frequency, second_frequency],
            amplitudes=[amplitude, second_amplitude],
            phases=[phase, second_phase],
            sample_rate=sample_rate,
            duration=duration
        )

    noisy_signal = add_noise(
        clean_signal,
        snr_db
    )

    return t, noisy_signal, signal_type
        
def generate_dataset(
        num_samples: int,
        sample_rate: int = 1000,
        duration: float = 1.0
):
    signals = []
    labels = []

    for _ in range(num_samples):
        _, signal, label = generate_random_signal(
            sample_rate=sample_rate,
            duration=duration
        )

        signals.append(signal)
        labels.append(label)

    return np.array(signals), np.array(labels)

if __name__ == "__main__":

    # t, signal, label = generate_random_signal()

    # print(f"Signal type: {label}")

    # plt.plot(t, signal)

    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title(f"Random {label.capitalize()} Signal")
    # plt.xlim(0, 0.3)

    # plt.grid(True)
    # plt.show()

    X, y = generate_dataset(
        num_samples=1000,
        sample_rate=1000,
        duration=1
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nLables:")
    print(y)
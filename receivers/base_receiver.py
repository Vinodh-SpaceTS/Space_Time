# receivers/base_receiver.py
from abc import ABC, abstractmethod

class BaseReceiver(ABC):
    @abstractmethod
    def measure(self, t: float, dt: float = 1.0) -> dict:
        """
        Take a measurement from the receiver at epoch t.
        
        Parameters:
          t (float): Current simulation time in seconds.
          dt (float): Sample interval in seconds.
          
        Returns:
          dict: A dictionary containing measurement fields:
                - "gnss_error" (float): Total GNSS time error (seconds)
                - "sat_count" (int): Number of visible satellites
                - "gps_count" (int): Number of visible GPS satellites
                - "galileo_count" (int): Number of visible Galileo satellites
                - "glonass_count" (int): Number of visible GLONASS satellites
                - "sbas_count" (int): Number of visible SBAS satellites
                - "sat_clock_error" (float): Satellite clock error component (seconds)
                - "propagation_error" (float): Propagation error component (seconds)
                - "measurement_noise" (float): Receiver measurement noise component (seconds)
                - "R" (float): Current measurement variance (seconds^2)
        """
        pass

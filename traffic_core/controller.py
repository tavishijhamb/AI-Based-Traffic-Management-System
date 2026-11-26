from dataclasses import dataclass

@dataclass
class SignalDecision:
    green_lane: str
    green_seconds: int
    emergency_priority: bool
    reason: str
    phase: str = "GREEN"
    active_lane: str = "North"
    remaining_seconds: float = 0.0

class AdaptiveSignalController:
    """
    Simulation-only adaptive traffic signal controller.

    Phases:
    GREEN -> YELLOW -> ALL_RED -> GREEN

    Uses video time rather than processing wall-clock time.
    """

    LANES = ["North", "South", "East", "West"]

    def __init__(
        self,
        min_green=8,
        max_green=45,
        seconds_per_vehicle=2,
        max_wait_cycles=3,
        yellow_seconds=3,
        all_red_seconds=1
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.seconds_per_vehicle = seconds_per_vehicle
        self.max_wait_cycles = max_wait_cycles

        self.yellow_seconds = yellow_seconds
        self.all_red_seconds = all_red_seconds

        self.wait_cycles = {k: 0 for k in self.LANES}

        self.current_lane = None
        self.next_lane = None

        self.phase = "GREEN"
        self.phase_elapsed = 0.0
        self.green_duration = 0

        self.emergency_lane = None
        self.emergency_active = False

        self.last_reason = "Initial signal selection"

    def select_lane(self, counts, emergency_lanes=None):

        emergency_lanes = emergency_lanes or []

        # Emergency priority is considered at a safe phase boundary.
        if emergency_lanes:
            lane = emergency_lanes[0]
            self.emergency_lane = lane
            self.emergency_active = True

        # Serve a waiting emergency vehicle first.
        if self.emergency_lane:
            lane = self.emergency_lane
            self.emergency_lane = None

            self.last_reason = (
                f"Emergency vehicle priority on {lane}"
            )

            return lane, True

        # Queue-weighted priority.
        scores = {
            lane: max(0, int(counts.get(lane, 0)))
            + 3 * self.wait_cycles[lane]
            for lane in self.LANES
        }

        lane = max(self.LANES, key=lambda x: scores[x])

        queue = max(0, int(counts.get(lane, 0)))

        duration = min(
            self.max_green,
            max(
                self.min_green,
                self.min_green
                + queue * self.seconds_per_vehicle
            )
        )

        self.green_duration = duration

        # Update waiting cycles.
        for direction in self.LANES:
            if direction == lane:
                self.wait_cycles[direction] = 0
            else:
                self.wait_cycles[direction] = min(
                    self.max_wait_cycles,
                    self.wait_cycles[direction] + 1
                )

        self.last_reason = (
            f"Weighted queue priority: {scores[lane]}"
        )

        return lane, False

    def decide(self, counts, emergency_lanes=None, dt=1/30):

        emergency_lanes = emergency_lanes or []

# Remember whether an emergency request is pending.
        if emergency_lanes:
            self.emergency_lane = emergency_lanes[0]
            self.emergency_active = True

        emergency_priority = self.emergency_active

        # First signal selection.
        if self.current_lane is None:
            self.current_lane, emergency_priority = (
                self.select_lane(counts, emergency_lanes)
            )

            self.phase = "GREEN"
            self.phase_elapsed = 0.0

        else:
            # Remember emergency vehicles for the next safe transition.
            if emergency_lanes:
                self.emergency_lane = emergency_lanes[0]
                self.emergency_active = True

            self.phase_elapsed += max(0.0, dt)

            if self.phase == "GREEN":

                if self.phase_elapsed >= self.green_duration:

                    self.phase = "YELLOW"
                    self.phase_elapsed = 0.0

                    # Select the next approach, but do not
                    # activate it until clearance is complete.
                    self.next_lane, _ = self.select_lane(counts, [])

                    # Preserve emergency priority indication.
                    emergency_priority = self.emergency_active

                    if self.emergency_active:
                        emergency_priority = True

            elif self.phase == "YELLOW":

                if self.phase_elapsed >= self.yellow_seconds:

                    self.phase = "ALL_RED"
                    self.phase_elapsed = 0.0

            elif self.phase == "ALL_RED":

                if self.phase_elapsed >= self.all_red_seconds:

                    self.current_lane = self.next_lane
                    self.next_lane = None

                    self.phase = "GREEN"
                    self.phase_elapsed = 0.0

                    # Calculate duration for the newly selected lane.
                    if self.emergency_active:
                        self.green_duration = self.max_green
                        emergency_priority = True
                        self.emergency_active = False
                        self.last_reason = (
                            f"Emergency priority activated for {self.current_lane}"
                        )
                    else:
                        queue = max(
                            0,
                            int(counts.get(self.current_lane, 0))
                        )

                        self.green_duration = min(
                            self.max_green,
                            max(
                                self.min_green,
                                self.min_green
                                + queue * self.seconds_per_vehicle
                            )
                        )

        # Calculate remaining phase time.
        if self.phase == "GREEN":
            remaining = self.green_duration - self.phase_elapsed
            active_lane = self.current_lane

        elif self.phase == "YELLOW":
            remaining = self.yellow_seconds - self.phase_elapsed
            active_lane = self.current_lane

        else:
            remaining = self.all_red_seconds - self.phase_elapsed
            active_lane = "None"

        return SignalDecision(
            green_lane=self.current_lane or "None",
            green_seconds=int(self.green_duration),
            emergency_priority=emergency_priority,
            reason=self.last_reason,
            phase=self.phase,
            active_lane=active_lane,
            remaining_seconds=max(0.0, remaining)
        )

def density_label(count):
    if count < 5:
        return "LOW"
    if count < 12:
        return "MEDIUM"
    return "HIGH"

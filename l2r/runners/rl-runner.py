# TODO check import
from environment.env import RacingEnv
from models.rl.buffer import ReplayBuffer

class ReinforcementLearningRunner():
    def __init__(self, agent_kwargs, env_kwargs, sim_kwargs):
        self.agent_kwargs = agent_kwargs
        self.env_kwargs = env_kwargs
        self.sim_kwargs = sim_kwargs
        # TODO: initialize environment
        self.env = RacingEnv(env_kwargs, sim_kwargs)
        # TODO: initialize agent
        self.agent = agent_kwargs.agent(agent_kwargs)
        # TODO: initialize visual encoder
        self.encoder = agent_kwargs.visual_encoder_type(agent_kwargs.visual_encoder_params)
        self.replay_buffer = ReplayBuffer()
        pass
    
    def _reset(self, random_pos=False):
        # reset the simulator
        env = self.env
        camera = 0
        self.file_logger(f"[trial episode] 1")
        while (np.mean(camera) == 0) | (np.mean(camera) == 255):
            self.file_logger(f"[trial episode] 2")
            obs = env.reset(random_pos=random_pos)
            self.file_logger(f"[trial episode] 3")
            (state, camera), _ = obs
            self.file_logger(f"[trial episode] 4")
        self.file_logger(f"[trial episode] 5")
        return camera, self._encode((state, camera)), state

    def _step(self, action):
        # sent the agent action to the simulator
        obs, reward, done, info = self.env.step(action)
        return obs[1], self._encode(obs), obs[0], reward, done, info

    def reset_episode(self, t):
        camera, feat, state = self._reset(random_pos=True)
        ep_ret, ep_len, self.metadata, experience = 0, 0, {}, []
        t_start = t + 1
        camera, feat, state2, r, d, info = self._step([0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start

    def train(self):
        best_ret, ep_ret, ep_len = 0, 0, 0

        self._reset(random_pos=True)
        camera, feat, state, r, d, info = self._step([0, 1])

        experience = []
        speed_dim = 1 if self.using_speed else 0
        assert (
            len(feat)
            == self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + speed_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = self.t_start
        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.t_start, self.cfg["total_steps"]):
            state_rep = self.encoder.encode(feat)
            a = self.agent.select_action(feat)

            # Step the env
            camera2, feat2, state2, r, d, info = self._step(a)

            # Check that the camera is turned on
            assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

            # Prevents the agent from getting stuck by sampling random actions
            # self.atol for SafeRandom and SPAR are set to -1 so that this condition does not activate
            if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                # self.file_logger("Sampling random action to get unstuck")
                a = self.env.action_space.sample()

                # Step the env
                camera2, feat2, state2, r, d, info = self._step(a)
                ep_len += 1

            state = state2
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.cfg["max_ep_len"] else d

            # Store experience to replay buffer
            if (not np.allclose(state2[15:16], state[15:16], atol=3e-1, rtol=0)) | (
                r != 0
            ):
                self.replay_buffer.store(feat, a, r, feat2, d)
            else:
                # print('Skip')
                skip = True

            if self.cfg["record_experience"]:
                recording = self.add_experience(
                    action=a,
                    camera=camera,
                    next_camera=camera2,
                    done=d,
                    env=env,
                    feature=feat,
                    next_feature=feat2,
                    info=info,
                    reward=r,
                    state=state,
                    next_state=state2,
                    step=t,
                )
                experience.append(recording)

                # quickly pass data to save thread
                # if len(experience) == self.save_batch_size:
                #    self.save_queue.put(experience)
                #    experience = []

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            feat = feat2
            state = state2  # in case we, later, wish to store the state in the replay as well
            camera = camera2  # in case we, later, wish to store the state in the replay as well

            # Update handling
            if (t >= self.cfg["update_after"]) & (t % self.cfg["update_every"] == 0):
                for j in range(self.cfg["update_every"]):
                    batch = self.replay_buffer.sample_batch(self.cfg["batch_size"])
                    self.agent.update(data=batch)
                    

            if (t + 1) % self.cfg["eval_every"] == 0:
                # eval on test environment
                val_returns = self.eval(t // self.cfg["eval_every"], env)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

            # End of trajectory handling
            if d or (ep_len == self.cfg["max_ep_len"]):
                self.metadata["info"] = info
                self.episode_num += 1
                msg = f"[Ep {self.episode_num }] {self.metadata}"
                self.file_logger(msg)
                self.log_train_metrics_to_tensorboard(ep_ret, t, t_start)

                # Quickly dump recently-completed episode's experience to the multithread queue,
                # as long as the episode resulted in "success"
                if self.cfg[
                    "record_experience"
                ]:  # and self.metadata['info']['success']:
                    self.file_logger("Writing experience")
                    self.save_queue.put(experience)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(t)
    
    def evaluate(self):
        pass
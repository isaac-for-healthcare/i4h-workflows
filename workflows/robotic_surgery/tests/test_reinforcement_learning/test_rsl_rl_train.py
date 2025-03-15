import os
import shutil
import subprocess
import tempfile
import unittest

from parameterized import parameterized

TEST_TRAIN_CASES = [
    [{"task": "Isaac-Reach-PSM-v0", "max_iterations": 5, "log_dir": "rsl_rl/Isaac-Reach-PSM-v0/test_run"}],
    [
        {
            "task": "Isaac-Lift-Needle-PSM-IK-Rel-v0",
            "max_iterations": 3,
            "log_dir": "rsl_rl/Isaac-Lift-Needle-PSM-IK-Rel-v0/test_run",
        }
    ],
]


class TestRSLRLTrain(unittest.TestCase):
    def setUp(self):
        # get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_path = os.path.join(current_dir, "..", "..", "scripts", "simulation", "scripts")
        self.train_script_path = os.path.join(scripts_path, "reinforcement_learning", "rsl_rl", "train.py")
        self.tmp_log_root_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_log_root_dir)

    @parameterized.expand(TEST_TRAIN_CASES)
    def test_rsl_rl_reach_psm_train(self, params):
        task = params["task"]
        max_iterations = params["max_iterations"]
        log_dir = os.path.join(self.tmp_log_root_dir, params["log_dir"])
        run_command = [
            "python",
            self.train_script_path,
            "--task",
            task,
            "--headless",
            "--max_iterations",
            str(max_iterations),
            "--log_dir",
            log_dir,
        ]
        # check if the process runs successfully
        subprocess.check_call(run_command)
        # check if the log directory exists
        self.assertTrue(os.path.exists(log_dir))
        # check get expected final iteration model weights
        final_model_weights = os.path.join(log_dir, f"model_{max_iterations-1}.pt")
        self.assertTrue(os.path.exists(final_model_weights))
        # check get expected subfolders: git and params
        self.assertTrue(os.path.exists(os.path.join(log_dir, "git")))
        self.assertTrue(os.path.exists(os.path.join(log_dir, "params")))
        # check get expected files: env.yaml, agent.yaml, env.pkl, agent.pkl in params subfolder
        self.assertTrue(os.path.exists(os.path.join(log_dir, "params", "env.yaml")))
        self.assertTrue(os.path.exists(os.path.join(log_dir, "params", "agent.yaml")))
        self.assertTrue(os.path.exists(os.path.join(log_dir, "params", "env.pkl")))
        self.assertTrue(os.path.exists(os.path.join(log_dir, "params", "agent.pkl")))


if __name__ == "__main__":
    unittest.main()

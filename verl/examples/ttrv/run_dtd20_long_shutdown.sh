#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$VERL_DIR"

mkdir -p logs

export TASK="${TASK:-dtd_20}"
export NO_GPU="${NO_GPU:-4}"
export EPISODE="${EPISODE:-20}"
export BACKBONE_PATH="${BACKBONE_PATH:-OpenGVLab/InternVL3-2B}"
export DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$VERL_DIR/data}"
export IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/datasets/dtd}"
export FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

export http_proxy="${http_proxy:-http://127.0.0.1:7892}"
export https_proxy="${https_proxy:-http://127.0.0.1:7892}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

TEST_FREQ="${TEST_FREQ:-25}"
SHUTDOWN_ON_EXIT="${SHUTDOWN_ON_EXIT:-1}"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-5}"

BASELINE_ACC="${BASELINE_ACC:-0.8996}"
COLLAPSE_ABS_FLOOR="${COLLAPSE_ABS_FLOOR:-0.80}"
COLLAPSE_DROP_FROM_BEST="${COLLAPSE_DROP_FROM_BEST:-0.05}"

BACKBONE_SAFE=$(echo "$BACKBONE_PATH" | tr '/' '_')
LOG_GLOB="logs/${TASK}_${BACKBONE_SAFE}_${EPISODE}e_"'*.log'

latest_log_file() {
  local -a candidates=()
  shopt -s nullglob
  candidates=(logs/${TASK}_${BACKBONE_SAFE}_${EPISODE}e_*.log)
  shopt -u nullglob
  if ((${#candidates[@]} == 0)); then
    return 1
  fi
  ls -t "${candidates[@]}" 2>/dev/null | head -n 1
}

write_summary() {
  local log_file="$1"
  local summary_file="${log_file%.log}.summary.txt"

  perl - "$log_file" "$BASELINE_ACC" "$COLLAPSE_ABS_FLOOR" "$COLLAPSE_DROP_FROM_BEST" >"$summary_file" <<'PERL'
use strict;
use warnings;

my ($log_file, $baseline_acc, $floor, $drop_limit) = @ARGV;
my @keys = (
  'val-aux/GPQA-TTT/acc/mean@1',
  'train/entropy',
  'actor/kl_loss',
  'response_length/mean',
  'perf/cpu_memory_used_gb',
);

open my $fh, '<', $log_file or die "cannot open $log_file: $!\n";
my @rows;
while (my $line = <$fh>) {
  next unless $line =~ /step:(\d+)/;
  my %row = (step => $1);
  my $has_metric = 0;
  for my $key (@keys) {
    if ($line =~ /\Q$key\E:([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/) {
      $row{$key} = $1;
      $has_metric = 1;
    }
  }
  push @rows, \%row if $has_metric;
}
close $fh;

@rows = sort { $a->{step} <=> $b->{step} } @rows;
my @val_rows = grep { exists $_->{'val-aux/GPQA-TTT/acc/mean@1'} } @rows;

print "log_file: $log_file\n";
print "generated_at: " . scalar(localtime) . "\n";
print "main_metric: val-aux/GPQA-TTT/acc/mean\@1\n";
print "do_not_compare_with: val-ttrl/label_accuracy\n";
print "reference_2_epoch_acc: $baseline_acc\n";
print "collapse_floor: $floor\n";
print "collapse_drop_from_best: $drop_limit\n";

if (@val_rows) {
  my $final = $val_rows[-1]->{'val-aux/GPQA-TTT/acc/mean@1'};
  my $final_step = $val_rows[-1]->{step};
  my $best = $final;
  my $best_step = $final_step;
  for my $row (@val_rows) {
    my $val = $row->{'val-aux/GPQA-TTT/acc/mean@1'};
    if ($val > $best) {
      $best = $val;
      $best_step = $row->{step};
    }
  }
  my $drop = $best - $final;
  my $status = 'CHECK_MANUALLY';
  if ($final < $floor || $drop > $drop_limit) {
    $status = 'SUSPECT_COLLAPSE';
  } elsif ($final >= 0.88 && $final <= 0.91) {
    $status = 'NO_OBVIOUS_COLLAPSE';
  } elsif ($final > 0.91) {
    $status = 'NO_OBVIOUS_COLLAPSE_HIGHER_THAN_REFERENCE';
  }

  printf "final_val_step: %s\n", $final_step;
  printf "final_val_acc: %.6f\n", $final;
  printf "best_val_step: %s\n", $best_step;
  printf "best_val_acc: %.6f\n", $best;
  printf "drop_from_best: %.6f\n", $drop;
  print "collapse_assessment: $status\n";
} else {
  print "collapse_assessment: NO_VALIDATION_METRIC_FOUND\n";
}

print "\nmetrics_by_logged_step\n";
print join("\t", 'step', @keys) . "\n";
for my $row (@rows) {
  print join("\t", map { exists $row->{$_} ? $row->{$_} : '' } ('step', @keys)) . "\n";
}
PERL

  echo "Summary written to: $summary_file"
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  echo
  echo "[cleanup] exit_code=$exit_code"
  if command -v ray >/dev/null 2>&1; then
    echo "[cleanup] stopping Ray..."
    ray stop --force >/dev/null 2>&1 || true
  fi

  local log_file=""
  if log_file=$(latest_log_file); then
    echo "[cleanup] latest log: $log_file"
    write_summary "$log_file" || echo "[cleanup] summary generation failed"
  else
    echo "[cleanup] no log found for pattern: $LOG_GLOB"
  fi

  case "$SHUTDOWN_ON_EXIT" in
    1|true|TRUE|yes|YES)
      if command -v shutdown >/dev/null 2>&1; then
        echo "[cleanup] scheduling shutdown in ${SHUTDOWN_DELAY_MINUTES} minutes."
        echo "[cleanup] cancel with: shutdown -c"
        shutdown -h +"$SHUTDOWN_DELAY_MINUTES" || true
      else
        echo "[cleanup] shutdown command not found; skip shutdown."
      fi
      ;;
    *)
      echo "[cleanup] SHUTDOWN_ON_EXIT=$SHUTDOWN_ON_EXIT, skip shutdown."
      ;;
  esac

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "[run] starting DTD long-run collapse check"
echo "[run] repo: $VERL_DIR"
echo "[run] task=$TASK backbone=$BACKBONE_PATH episode=$EPISODE no_gpu=$NO_GPU test_freq=$TEST_FREQ"
echo "[run] data=$DATA_LOCAL_DIR image_root=$IMAGE_ROOT"
echo "[run] expected log pattern: $LOG_GLOB"
echo "[run] shutdown_on_exit=$SHUTDOWN_ON_EXIT delay=${SHUTDOWN_DELAY_MINUTES}m"

set +e
bash "$SCRIPT_DIR/run.sh" \
  trainer.val_before_train=False \
  trainer.test_freq="$TEST_FREQ" \
  trainer.resume_mode=disable \
  "$@"
run_status=$?
set -e

exit "$run_status"

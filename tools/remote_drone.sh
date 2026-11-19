#!/usr/bin/env bash
# Work with the drone through the second laptop, from this machine.
#
# Topology:
#     [this laptop] --cable 192.168.50.0/24-- [second laptop] --Wi-Fi 5GHz-- [drone 10.42.0.1]
#
# The second laptop loses its home Wi-Fi when it joins the drone AP, so the
# cable is the only path. Nothing here routes IP through the second laptop:
# SSH port forwarding gives this machine a local port that tunnels straight to
# the drone. That keeps the VPN and the home Wi-Fi untouched.
#
# Usage:
#   tools/remote_drone.sh status          # is the link up? is the drone reachable?
#   tools/remote_drone.sh forward         # open tunnels (background, prints PIDs)
#   tools/remote_drone.sh stop            # close the tunnels
#   tools/remote_drone.sh pull <dir>      # copy drone frames to samples/frames/
#
# Environment overrides:
#   SECOND_HOST (default 192.168.50.2)  SECOND_USER (default: same as local user)
#   DRONE_IP    (default 10.42.0.1)

set -uo pipefail

SECOND_HOST="${SECOND_HOST:-192.168.50.2}"
SECOND_USER="${SECOND_USER:-$(id -un)}"
DRONE_IP="${DRONE_IP:-10.42.0.1}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed22519}"
# The local key file is named id_ed22519 (an old typo), so ssh will not pick it
# up automatically — it must be passed with -i.
SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="${TMPDIR:-/tmp}/v2m-drone-tunnels"

die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

remote() {
    ssh "${SSH_OPTS[@]}" "${SECOND_USER}@${SECOND_HOST}" "$@"
}

eth_iface() {
    nmcli -t -f DEVICE,TYPE device status \
        | awk -F: '$2=="ethernet" && $3!="unmanaged" {print $1; exit}'
}

# The VPN answers ICMP to arbitrary addresses, so a successful ping proves
# nothing: with the cable unplugged `ping 192.168.50.2` still returns replies
# from 172.19.0.1 via happ-xray. Always confirm the kernel would really use the
# cable before trusting any reachability test.
link_iface() {
    ip route get "$SECOND_HOST" 2>/dev/null \
        | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}'
}

is_vpn_iface() {
    case "$1" in
        happ-xray|tun*|wg*|utun*) return 0 ;;
        *) return 1 ;;
    esac
}

# Local port -> drone port. 9999/9090 are the IDE and Jupyter, 22 is ssh.
TUNNEL_PORTS=(9999 9090 22)

cmd_status() {
    local eth carrier used
    eth=$(eth_iface)
    printf 'Второй ноут : %s (%s)\n' "$SECOND_HOST" "$SECOND_USER"
    printf 'Интерфейс   : %s\n' "${eth:-не найден}"

    if [[ -z "$eth" ]]; then
        printf 'Кабель      : НЕТ Ethernet-интерфейса\n'
        return 1
    fi

    carrier=$(cat "/sys/class/net/${eth}/carrier" 2>/dev/null || echo 0)
    used=$(link_iface)
    printf 'Маршрут     : %s\n' "${used:-нет}"

    if is_vpn_iface "$used"; then
        printf 'Кабель      : трафик уходит в VPN (%s), а не в кабель\n' "$used"
        printf '\nКабель не воткнут либо адрес не настроен. На втором ноуте:\n'
        printf '  sudo tools/second_laptop_setup.sh\n'
        return 1
    fi

    if [[ "$carrier" != "1" ]]; then
        printf 'Кабель      : не подключён (carrier=0)\n'
        printf '\nВоткните кабель и запустите на втором ноуте:\n'
        printf '  sudo tools/second_laptop_setup.sh\n'
        return 1
    fi

    if [[ "$used" != "$eth" ]]; then
        printf 'Кабель      : маршрут идёт через %s, а не через %s\n' "$used" "$eth"
        return 1
    fi

    printf 'Кабель      : OK\n'

    if ! remote true 2>/dev/null; then
        printf 'SSH         : НЕ РАБОТАЕТ\n'
        printf '\nНа втором ноуте нужен sshd и ваш ключ:\n'
        printf '  sudo tools/second_laptop_setup.sh --key "%s"\n' "$(cat "${SSH_KEY}.pub" 2>/dev/null || echo '<ключ>')"
        return 1
    fi
    printf 'SSH         : OK\n'

    local wifi
    wifi=$(remote "nmcli -t -f GENERAL.CONNECTION device show 2>/dev/null | grep -m1 wifi -A0 || true")
    printf 'Сеть ноута  : %s\n' "${wifi:-неизвестно}"

    if remote "ping -c1 -W2 ${DRONE_IP} >/dev/null 2>&1"; then
        printf 'Дрон        : доступен (%s)\n' "$DRONE_IP"
    else
        printf 'Дрон        : НЕ доступен — на втором ноуте подключитесь к точке дрона\n'
        return 1
    fi

    local port
    for port in "${TUNNEL_PORTS[@]}"; do
        if remote "timeout 3 bash -c 'exec 3<>/dev/tcp/${DRONE_IP}/${port}' 2>/dev/null"; then
            printf '  порт %-5s на дроне открыт\n' "$port"
        else
            printf '  порт %-5s на дроне закрыт\n' "$port"
        fi
    done
    return 0
}

cmd_forward() {
    mkdir -p "$PID_DIR"
    cmd_status >/dev/null 2>&1 || die "сначала добейтесь зелёного status"

    local port
    for port in "${TUNNEL_PORTS[@]}"; do
        if ss -ltn 2>/dev/null | grep -q ":${port} "; then
            printf 'Локальный порт %s занят — пропускаю (туннель уже есть?)\n' "$port"
            continue
        fi
        ssh "${SSH_OPTS[@]}" -fN -L "${port}:${DRONE_IP}:${port}" \
            "${SECOND_USER}@${SECOND_HOST}" \
            || { printf 'Не удалось поднять туннель на %s\n' "$port" >&2; continue; }
        printf '%s\n' "$port" > "${PID_DIR}/${port}.port"
        printf 'Туннель %s -> %s:%s открыт\n' "$port" "$DRONE_IP" "$port"
    done

    printf '\nТеперь локально:\n'
    printf '  IDE:     http://localhost:9999/?folder=/home/pioneermini/workspace\n'
    printf '  Jupyter: http://localhost:9090/\n'
    printf '  ssh:     ssh -p 22 pioneer@localhost\n'
}

cmd_stop() {
    local stopped=0 port
    for port in "${TUNNEL_PORTS[@]}"; do
        # -fN tunnels leave no useful pidfile, so match the exact -L spec.
        if pkill -f -- "-L ${port}:${DRONE_IP}:${port}" 2>/dev/null; then
            printf 'Туннель %s закрыт\n' "$port"
            stopped=1
        fi
        rm -f "${PID_DIR}/${port}.port"
    done
    (( stopped )) || printf 'Открытых туннелей не найдено.\n'
}

cmd_pull() {
    local remote_dir="${1:-/home/pioneermini/workspace}"
    local dest="${2:-${REPO_ROOT}/samples/frames}"
    mkdir -p "$dest"

    printf 'Копирую %s:%s -> %s\n' "$SECOND_HOST" "$remote_dir" "$dest"
    # The drone is not reachable from here directly, so hop through the second
    # laptop, which does have the drone route.
    scp "${SSH_OPTS[@]}" -o ProxyCommand="ssh ${SSH_OPTS[*]} -W ${DRONE_IP}:22 ${SECOND_USER}@${SECOND_HOST}" \
        -r "pioneer@${DRONE_IP}:${remote_dir}" "$dest" \
        || die "не удалось скопировать (ssh на дроне может быть выключен)"
    printf 'Готово. Проверьте: ls %s\n' "$dest"
}

case "${1:-status}" in
    status|check) cmd_status ;;
    forward|up)   cmd_forward ;;
    stop|down)    cmd_stop ;;
    pull)         shift; cmd_pull "$@" ;;
    -h|--help|help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0" ;;
    *) die "неизвестная команда: $1 (status | forward | stop | pull)" ;;
esac
